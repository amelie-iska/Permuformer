# Permuformer
## Random Order Transformer

The codebase is organized as follows: 
```
random_decoding_order_transformer/
│
├── data/
│   ├── dataset.py
│   └── dataloader.py
│
├── models/
│   ├── transformer.py
│   └── modules/
│       ├── attention.py
│       ├── embedding.py
│       ├── feedforward.py
│       └── positional_encoding.py
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   └── utils.py
│
├── train.py
└── eval.py
```

This is a random order decoding autoregressive transformer, trained on human proteins up to length 200. 