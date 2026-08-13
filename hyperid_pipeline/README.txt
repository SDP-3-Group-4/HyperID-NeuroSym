HYPER-ID pipeline

1. Plain CLIP baseline:
   python hyperid_clip_baseline.py path/to/dog.jpg

2. Grounded trait + SPARQL reasoning:
   python hyperid_reasoning.py path/to/dog.jpg

3. Oxford sanity evaluation:
   python hyperid_oxford_eval.py --limit-per-breed 10

The reasoning script now follows the supplied SPARQL Query #3:
VALUES ?trait -> ?breed :hasTrait ?trait -> COUNT -> GROUP BY -> ORDER BY DESC.

Required project files beside these scripts:
  breed_trait_ontology.ttl
  trained_model/multitask_trait_model.pt
  trained_model/label_encoders.pkl

Local/mixed-breed collection and local phenotype ontology updates are not included yet.
