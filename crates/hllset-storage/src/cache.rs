//! Caching storage wrapper — redundancy tier with LRU + TTL eviction.
//!
//! Wraps any `Storage` backend. Cache is a safe redundancy layer:
//! elements are deterministic (content-addressed), so eviction
//! only removes a LOCAL copy — the canonical data remains in IPFS
//! or on peer nodes.

use crate::storage::{Result, Storage, StorageError};
use std::collections::{BTreeMap, VecDeque};

/// Cache entry with access tracking.
#[derive(Clone)]
struct CacheEntry {
    data: Vec<u8>,
    last_access: std::time::Instant,
    insert_time: std::time::Instant,
}

/// LRU cache wrapping a storage backend.
///
/// Hit → promotes entry (moves to front of LRU queue).
/// Miss → fetches from inner storage, caches result.
/// Eviction → removes least-recently-used or TTL-expired entries.
#[derive(Clone)]
pub struct CacheStorage<S: Storage + Clone + 'static> {
    inner: S,
    cache: std::rc::Rc<std::cell::RefCell<BTreeMap<String, CacheEntry>>>,
    lru: std::rc::Rc<std::cell::RefCell<VecDeque<String>>>,
    max_entries: usize,
    ttl: std::time::Duration,
}

impl<S: Storage + Clone + 'static> CacheStorage<S> {
    /// Wrap a storage backend with an LRU cache.
    ///
    /// - `max_entries`: maximum cached entries before eviction.
    /// - `ttl`: time-to-live for cache entries (evicted on access if expired).
    pub fn new(inner: S, max_entries: usize, ttl: std::time::Duration) -> Self {
        Self {
            inner,
            cache: Default::default(),
            lru: Default::default(),
            max_entries,
            ttl,
        }
    }

    /// Number of entries currently cached.
    pub fn cache_size(&self) -> usize {
        self.cache.borrow().len()
    }

    /// Evict entries from cache (LRU policy: evict the tail).
    fn evict_lru(&self, count: usize) {
        let mut lru = self.lru.borrow_mut();
        let mut cache = self.cache.borrow_mut();
        for _ in 0..count {
            if let Some(key) = lru.pop_back() {
                cache.remove(&key);
            }
        }
    }

    /// Touch an entry (move to front of LRU).
    fn touch(&self, key: &str) {
        let mut lru = self.lru.borrow_mut();
        if let Some(pos) = lru.iter().position(|k| k == key) {
            lru.remove(pos);
        }
        lru.push_front(key.to_string());
    }
}

impl<S: Storage + Clone + 'static> Storage for CacheStorage<S> {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        // Store to inner (canonical)
        self.inner.put(key, data)?;

        // Cache locally
        let mut cache = self.cache.borrow_mut();
        cache.insert(
            key.to_string(),
            CacheEntry {
                data: data.to_vec(),
                last_access: std::time::Instant::now(),
                insert_time: std::time::Instant::now(),
            },
        );
        drop(cache);

        self.touch(key);

        // Evict if over capacity
        if self.cache_size() > self.max_entries {
            self.evict_lru(self.cache_size() - self.max_entries);
        }

        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Check cache first
        let cached: Option<Vec<u8>> = {
            let cache = self.cache.borrow();
            cache.get(key).and_then(|entry| {
                if entry.insert_time.elapsed() < self.ttl {
                    Some(entry.data.clone())
                } else {
                    None // expired
                }
            })
        };
        if let Some(data) = cached {
            self.touch(key);
            return Ok(Some(data));
        }

        // Cache miss or TTL expired → fetch from inner
        match self.inner.get(key)? {
            Some(data) => {
                // Cache the fresh data
                let entry = CacheEntry {
                    data: data.clone(),
                    last_access: std::time::Instant::now(),
                    insert_time: std::time::Instant::now(),
                };
                self.cache.borrow_mut().insert(key.to_string(), entry);
                self.touch(key);

                // Evict if needed
                if self.cache_size() > self.max_entries {
                    self.evict_lru(self.cache_size() - self.max_entries);
                }

                Ok(Some(data))
            }
            None => Ok(None),
        }
    }

    fn has(&self, key: &str) -> Result<bool> {
        // Check cache first (fast path)
        if self.cache.borrow().contains_key(key) {
            return Ok(true);
        }
        self.inner.has(key)
    }

    fn delete(&self, key: &str) -> Result<bool> {
        self.cache.borrow_mut().remove(key);
        let mut lru = self.lru.borrow_mut();
        if let Some(pos) = lru.iter().position(|k| k == key) {
            lru.remove(pos);
        }
        self.inner.delete(key)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        // Return union of cached + inner keys
        let mut keys: std::collections::BTreeSet<String> = self
            .cache
            .borrow()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        for k in self.inner.list(prefix)? {
            keys.insert(k);
        }
        Ok(keys.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryStorage;
    use std::time::Duration;

    #[test]
    fn test_cache_hit() {
        let inner = MemoryStorage::new();
        let cache = CacheStorage::new(inner, 100, Duration::from_secs(60));

        cache.store("h:test", b"cached").unwrap();
        let data = cache.load("h:test").unwrap().unwrap();
        assert_eq!(data, b"cached");
    }

    #[test]
    fn test_cache_eviction_on_capacity() {
        let inner = MemoryStorage::new();
        let cache = CacheStorage::new(inner, 2, Duration::from_secs(60));

        cache.store("h:a", b"1").unwrap();
        cache.store("h:b", b"2").unwrap();
        cache.store("h:c", b"3").unwrap(); // should evict "h:a" (oldest LRU)

        assert!(cache.cache_size() <= 2);
    }

    #[test]
    fn test_ttl_expiry_falls_through_to_inner() {
        let inner = MemoryStorage::new();
        let cache = CacheStorage::new(inner.clone(), 100, Duration::from_millis(1));

        cache.store("h:x", b"test").unwrap();
        std::thread::sleep(Duration::from_millis(5));

        // Cache entry expired, should fetch from inner
        let data = cache.load("h:x").unwrap().unwrap();
        assert_eq!(data, b"test");
    }

    #[test]
    fn test_delete_clears_cache() {
        let inner = MemoryStorage::new();
        let cache = CacheStorage::new(inner, 100, Duration::from_secs(60));

        cache.store("h:del", b"x").unwrap();
        assert!(cache.exists("h:del").unwrap());
        cache.delete("h:del").unwrap();
        assert!(!cache.exists("h:del").unwrap());
    }
}
