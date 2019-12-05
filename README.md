# Mal2vec
Word2Vec is one of the most successful and popular technologies for Natural Language Processing. It facilitates the understanding of the semantics of words using their context. Many other domains adopted the Word2Vec approach and used embedding of domain objects in Euclidean spaces for distance calculation, clustering, visualization and more.

Mal2Vec is a Word2Vec-based framework for analytics of security incidents that helps the analyst understand the contextual relations between attack vectors, and thus to understand better attack flows. The tool looks at malicious web request as words and at sequences of malicious web requests as sentences, and applies a variant of Word2Vec to embed the attack vectors in Euclidean space and to analyze their contextual relations. Using this approach, the analyst can get better understanding of the attack flows, e.g., he can see which attack vectors tend to come together.

While we developed Mal2Vec to improve our understanding of web attack based on analysis of security events of Web Application Firewall (WAF), we also provide an easy customization flow that will make it useful for analytics of other cyber-attack data.

**Presented at:** [BlackHat Europe 2019](https://www.blackhat.com/eu-19/arsenal/schedule/#malvec-wordvec-variant-for-analytics-of-web-attacks-17713) and [OWASP AppSec Global 2019](https://www.youtube.com/watch?v=a1fYsYfxqzo)

## Getting Started

### Prerequisites

```
Jupyter Notebook
```

### Installing

Clone the project

```
git clone https://github.com/imperva/mal2vec.git
```

Run Jupyter Notebook

```
jupyter-notebook --notebook-dir={path_to_project}
```

## Executing Example

1) Open the example notebook under `examples/WAF example.ipynb`
2) Execute the first 2 cells (Imports & Load CSV)
3) Load the data file `data/WAF.gz` by clicking 'Load CSV'
4) Execute the next 2 cells (Map columns & Select additional grouping columns). The correct columns are already selected.
5) Create senteces, Prepare dataset, Train and Evaluate the model by executing each of the following cells and clicking 'Start'
6) The evaluation cell will provide a visual 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details