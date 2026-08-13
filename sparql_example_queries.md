# Example SPARQL Queries — Breed & Trait Ontology

Run these against `breed_trait_ontology.ttl` — either in Protégé's built-in SPARQL query tab, or via `rdflib` in Python for the live demo script. All queries use this prefix:

```sparql
PREFIX : <http://straycare.org/ontology/breed#>
```

## 1. Which breeds have a given trait?
Sanity-checks the ontology and is the basic building block of the reasoning layer.

```sparql
SELECT ?breed WHERE {
  ?breed :hasEarShape :FloppyEars .
}
```

## 2. Which breeds share a breed group with a given breed?
Used as a fallback when a single-breed match is low-confidence — fall back to a group-level answer instead of a wrong specific one.

```sparql
SELECT ?breed WHERE {
  :Beagle :belongsToGroup ?group .
  ?breed :belongsToGroup ?group .
}
```

## 3. Rank candidate breeds by trait overlap with a detected trait profile
This is the actual reasoning behind the ancestry-estimate output: given the traits a photo's embeddings were classified as having, rank breeds by how many traits match.

```sparql
SELECT ?breed (COUNT(?trait) AS ?sharedTraits) WHERE {
  VALUES ?trait { :SemiErectEars :ShortCoat :MediumSize }
  ?breed :hasTrait ?trait .
}
GROUP BY ?breed
ORDER BY DESC(?sharedTraits)
```

In the running pipeline, the `VALUES` line is generated dynamically from that image's per-trait predictions rather than hardcoded — this query is the template the reasoning-layer code fills in.

## 4. Consistency check (live reasoner demo)
Because `hasSizeClass`, `hasCoatType`, etc. are declared `owl:FunctionalProperty`, asserting a second, conflicting size class for the same breed and then running Protégé's reasoner (HermiT or Pellet) will surface an inconsistency. This is a good ~30-second live demo that the ontology is doing actual reasoning work, not just storing a table of facts — worth showing your supervisor directly in Protégé.
