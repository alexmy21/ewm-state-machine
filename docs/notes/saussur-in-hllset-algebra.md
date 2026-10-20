# Notes for fine tuning Saussure in HLLSet Algebra

## Part 1

- Definition 2.2.
  - "A map $\text{Ing} : \text{Tok} \to \text{Lat}$ from tokens to nodes is an \textbf{IICA
morphism} if it is (i) \emph{idempotent}, read as deterministic: the same token
always maps to the same node" - the morphism Ing is a map from tokens to bit positions in HLLSets (nodes in lattice). Bita are HLLSet agnostic, so, we need to check this in all occurences;
  - Similar for materialization: from HLLSet to tokens through disambiguation of each bit = 1 in the given HLLSet;
- Reading 3.1.(The linguistic sign). It's formally accurate, but examples from Saussure use words (tokens) and this can create wrong impression that sign is token - it's HLLSet even in case of single bit activated.
- Remark 3.9. Would be nice to point to the fact Saussure's linearity compacted into HLLSet that breaks natural order of tokens, but n-gram encoding that we are using in reading ordered streams allows us to restore original order using de Bruijn algorithm. (n-gram is micro-context).
- 4.3. The paradox resolved: two layers - The H(t) = (S(t), H(t-1), D,R,N) is the general solution for all related to immutability - mutability discussions. Yes, there 2 layers in commit historical stack. Things are changing by departing from the past, new substitute old without overwriting.
