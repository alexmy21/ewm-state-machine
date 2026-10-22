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

## Part 2

- Apply notes from Part 1 to address clarification of morphisms in chapter 3 (like tokens to bits in HLLSet, not to HLLSet);
- 3.3 Realities - "hard key" as matching HLLSets using bit-vectors, or HLLSet ring base matching HLLSets by matching liniar compositions on ring base HLLSets. Soft keys are always defined by matching bit-vetors of HLLSets.
- 4.1 Thought and sound coupled - technically is correct however wording is a little bit convoluted.
- Theorem 4.7 (Value is a complete invariant on the span) - is very well stated but may be confusing and needed some clarification. Ring base is actually ontology in the same sense as labels. Ontology is not about semantical correctness (this is for people) but about relations in the lattice. From this point of view human readable ontology presented by meaningful labels and ring base are the same - they are isomorphic. And just because any collection of HLLSet has natural presentation as HLLSet -> value (conceptual as well) is HLLSet with its provenance.

## Part 3, 4 and 5

Is where we picked up our style, so it would be up to you and if changes in part 1 and 2 needed tunning there
