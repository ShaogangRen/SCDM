/* Author-editable English page content. Values are manuscript transcriptions, not new experiments. */
window.SCDM = {
  "title": "SCDM: Scalable Causal Discovery in<br class=\"desktop-break\"> Nonlinear Temporal Systems<br class=\"desktop-break\"> with Meta-Learning",
  "authors": [
    {
      "name": "Jingbo Wang",
      "url": null
    },
    {
      "name": "Kegeng Tang",
      "url": null
    },
    {
      "name": "Zihao Wang",
      "url": "https://drzihao.wang/"
    },
    {
      "name": "Shaogang Ren",
      "url": "https://shaogangren.github.io/",
      "corresponding": true
    }
  ],
  "abstract": "Causal discovery from high-dimensional nonlinear time series requires estimating a quadratically growing set of directed relations. Target-wise model fitting and repeated conditional testing can make this task computationally expensive. We propose SCDM, a shared-parameter meta-learning framework that treats each target variable as a related prediction task. Each episode adapts a shared temporal predictor and structural parameters on support data, then updates the shared initialization using query losses from sampled targets. This accumulates target-specific information without processing all targets in every episode. Post-training readout produces continuous directed scores for graph evaluation. We provide a finite-error recovery principle for fully observed delayed systems, conditional on a population risk margin and sufficiently small statistical, meta-optimization, and readout errors. Controlled benchmarks show competitive, dataset-dependent recovery. High-dimensional simulations examine runtime–accuracy tradeoffs, while LargeST provides real-world feasibility, forecasting, and spatial-plausibility diagnostics. Together, these evaluations assess scalable dependency estimation while retaining the assumptions required for causal interpretation.",
  "methods": [
    {
      "title": "Sample target tasks",
      "subtitle": "Predict each target from the lagged system",
      "heading": "A variable becomes a task.",
      "copy": "Target i is predicted from the lagged histories of the observed variables. Each episode samples a target subset and support/query windows. Across episodes, tasks contribute to the shared initialization.",
      "equation": "H̃<sup>(i)</sup><sub>t,j,ℓ</sub> = W<sub>j,i</sub> x<sup>(j)</sup><sub>t−ℓ</sub>",
      "blocks": [
        "Lagged history",
        "Target subset",
        "Support / query"
      ]
    },
    {
      "title": "Adapt on support",
      "subtitle": "Create temporary task-local copies",
      "heading": "Adaptation is temporary.",
      "copy": "All sampled tasks start from the same Θ = {θ, W}. Support gradients adapt a local copy for K steps. There is no persistent private head or target-specific model.",
      "equation": "Θ′<sub>i</sub> = Adapt<sub>K</sub>(Θ, D<sup>sup</sup><sub>i</sub>)",
      "blocks": [
        "Shared Θ",
        "Support updates",
        "Temporary Θ′ᵢ"
      ]
    },
    {
      "title": "Update on query",
      "subtitle": "Aggregate tasks into one initialization",
      "heading": "Query losses update shared parameters.",
      "copy": "Evaluate the adapted copies on query data and average their losses. A first-order meta-update changes the persistent shared predictor and structural matrix. Discard the adapted copies after the episode.",
      "equation": "Θ ← Θ − β · (1/B) ∑<sub>i∈𝓑</sub> ∇<sub>Θ′ᵢ</sub> L<sup>qry</sup><sub>i</sub>(Θ′<sub>i</sub>)",
      "blocks": [
        "Query losses",
        "Task average",
        "Updated shared Θ"
      ]
    },
    {
      "title": "Read out the graph",
      "subtitle": "Separate learning from score extraction",
      "heading": "Convert the trained model into scores.",
      "copy": "Direct structural scoring uses |Wⱼ,ᵢ|. A separate source-isolated predictive readout retains only one source history and compares against a zero-input baseline. The two rules have dataset-dependent behavior and should be distinguished.",
      "equation": "S<sub>j→i</sub> ∈ ℝ  ·  source j → target i",
      "blocks": [
        "Trained predictor",
        "Graph readout",
        "Directed scores"
      ]
    }
  ],
  "datasets": [
    {
      "key": "nc8",
      "name": "NC8",
      "takeaway": "SCDM ranks second in both AUROC and AUPRC on NC8. UnCLe has the highest reported means."
    },
    {
      "key": "eeg",
      "name": "Simulated EEG",
      "takeaway": "SCDM has the highest reported AUPRC on simulated EEG. GVAR has the highest AUROC, showing that the ranking depends on the metric."
    },
    {
      "key": "fmri",
      "name": "Simulated fMRI",
      "takeaway": "SCDM has the highest reported AUROC and AUPRC on simulated fMRI among the compared methods. This is a controlled simulation with known ground truth."
    }
  ],
  "results": [
    {
      "name": "UnCLe",
      "nc8": {
        "auroc": [
          0.947,
          0.014
        ],
        "auprc": [
          0.756,
          0.08
        ]
      },
      "eeg": {
        "auroc": [
          0.571,
          0.008
        ],
        "auprc": [
          0.155,
          0.008
        ]
      },
      "fmri": {
        "auroc": [
          0.661,
          0.012
        ],
        "auprc": [
          0.268,
          0.006
        ]
      }
    },
    {
      "name": "PCMCI",
      "nc8": {
        "auroc": [
          0.909,
          0.003
        ],
        "auprc": [
          0.594,
          0.022
        ]
      },
      "eeg": {
        "auroc": [
          0.743,
          0.012
        ],
        "auprc": [
          0.39,
          0.018
        ]
      },
      "fmri": {
        "auroc": [
          0.569,
          0.012
        ],
        "auprc": [
          0.16,
          0.01
        ]
      }
    },
    {
      "name": "TCDF",
      "nc8": {
        "auroc": [
          0.74,
          0.019
        ],
        "auprc": [
          0.415,
          0.02
        ]
      },
      "eeg": {
        "auroc": [
          0.685,
          0.005
        ],
        "auprc": [
          0.224,
          0.007
        ]
      },
      "fmri": {
        "auroc": [
          0.585,
          0.019
        ],
        "auprc": [
          0.178,
          0.02
        ]
      }
    },
    {
      "name": "GVAR",
      "nc8": {
        "auroc": [
          0.75,
          0.013
        ],
        "auprc": [
          0.413,
          0.041
        ]
      },
      "eeg": {
        "auroc": [
          0.75,
          0.004
        ],
        "auprc": [
          0.344,
          0.019
        ]
      },
      "fmri": {
        "auroc": [
          0.619,
          0.017
        ],
        "auprc": [
          0.219,
          0.019
        ]
      }
    },
    {
      "name": "CUTS+",
      "nc8": {
        "auroc": [
          0.795,
          0.025
        ],
        "auprc": [
          0.354,
          0.035
        ]
      },
      "eeg": {
        "auroc": [
          0.589,
          0.013
        ],
        "auprc": [
          0.16,
          0.007
        ]
      },
      "fmri": {
        "auroc": [
          0.532,
          0.009
        ],
        "auprc": [
          0.141,
          0.009
        ]
      }
    },
    {
      "name": "cMLP",
      "nc8": {
        "auroc": [
          0.753,
          0.002
        ],
        "auprc": [
          0.524,
          0.006
        ]
      },
      "eeg": {
        "auroc": [
          0.656,
          0.014
        ],
        "auprc": [
          0.258,
          0.024
        ]
      },
      "fmri": {
        "auroc": [
          0.624,
          0.013
        ],
        "auprc": [
          0.24,
          0.02
        ]
      }
    },
    {
      "name": "SCDM",
      "nc8": {
        "auroc": [
          0.916,
          0.015
        ],
        "auprc": [
          0.704,
          0.035
        ]
      },
      "eeg": {
        "auroc": [
          0.717,
          0.037
        ],
        "auprc": [
          0.476,
          0.067
        ]
      },
      "fmri": {
        "auroc": [
          0.867,
          0.006
        ],
        "auprc": [
          0.728,
          0.019
        ]
      }
    }
  ],
  "gallery": [
    {
      "tab": "fMRI recovery",
      "type": "Controlled simulation · known graph",
      "title": "Recovering atlas-grounded interactions",
      "file": "assets/fmri.webp",
      "width": 1596,
      "height": 1028,
      "caption": "Schaefer-100 ROI layout with Yeo-network labels; network-sorted directed ground truth; SCDM scores in the same source–target order; and representative recovered interactions mapped to ROI space. Original Figure 2 extracted from the manuscript.",
      "boundary": "This is a neuroimaging-inspired simulation, not causal validation on recordings from human participants. Displayed interactions are illustrative."
    },
    {
      "tab": "Benchmark construction",
      "type": "Simulation design · anatomical metadata",
      "title": "A benchmark with explicit ground truth",
      "file": "assets/fmri-benchmark.webp",
      "width": 3464,
      "height": 2231,
      "caption": "Atlas-based ROI layout, network-sorted ground-truth graph, neural-to-BOLD observation model, and masked ROIs used for a separate latent-view evaluation. This figure describes the simulation setup, not recovered scores.",
      "boundary": "The latent-view setting is a controlled diagnostic. It does not establish identifiability under arbitrary unmeasured confounding."
    },
    {
      "tab": "LargeST traffic",
      "type": "Real-world data · spatial plausibility",
      "title": "Directed structure across traffic sensors",
      "file": "assets/traffic.webp",
      "width": 3457,
      "height": 1177,
      "caption": "LargeST-GLA sensor layout, representative high-scoring local directed relations, and geographic-distance distributions for random pairs and selected relations at different training stages. Map imagery: © OpenStreetMap contributors.",
      "boundary": "LargeST does not provide complete directed causal ground truth. Nearby inferred relations support spatial plausibility, not verification of individual causal edges."
    }
  ],
  "bibtex": "@inproceedings{wang2026scdm,\n  title     = {SCDM: Scalable Causal Discovery in Nonlinear Temporal Systems with Meta-Learning},\n  author    = {Wang, Jingbo and Tang, Kegeng and Wang, Zihao and Ren, Shaogang},\n  booktitle = {Advances in Neural Information Processing Systems},\n  year      = {2026},\n  url       = {https://shaogangren.github.io/SCDM/}\n}"
};
