# XRec: Large Language Models for Explainable Recommendation

<img src='XRec_cover.png' />

PyTorch implementation for [XRec: Large Language Models for Explainable Recommendation](http://arxiv.org/abs/2406.02377)

 >**XRec: Large Language Models for Explainable Recommendation**  
 >Qiyao Ma, Xubin Ren, Chao Huang*\
 >*Preprint 2024*

-----

This paper presents a model-agnostic framework, **XRec**, that integrates the graph-based collaborative filtering framework with Large Language Models (LLMs) to generate comprehensive explanations for recommendations. By leveraging the inherent collaborative user-item relationships and harnessing the powerful textual generation capabilities of LLMs, XRec establishes a strong connection between collaborative signals and language semantics through the utilization of a Mixture of Experts (MoE) adapter.

<p align="center">
<img src="XRec.png" alt="XRec" />
</p>

## Environment

Run the following command to install dependencies:

```
pip install -r requirements.txt
```

## Datasets

We utilize three public datasets: Amazon-books `amazon`, Google-reviews `google`, Yelp `yelp`.User/item profile and explanations for each dataset are provided in `data/{dataset}/data.json`.To generate user/item profile and explanations from scratch, enter your **OpenAI API Key** in line 7 of these files: `generation/{item_profile/user_profile/explanation}/generate_{profile/exp}.py`.

- **Item Profile Generation**:
  ```
  python generation/item_profile/generate_profile.py
  ```
- **User Profile Generation**:
  ```
  python generation/user_profile/generate_profile.py
  ```
- **Explanation Generation**:
  ```
  python generation/explanation/generate_exp.py
  ```

## Usage

Each of the below commands can be run independently, since the finetuned LLM and generated explanations are provided within the data. Prepare your **Hugging Face User Access Token** for downloading Llama 2 model.

- To finetune the LLM from scratch:
  ```
  python explainer/main.py --mode finetune --dataset {dataset}
  ```
- To generate explanations:
  ```
  python explainer/main.py --mode generate --dataset {dataset}
  ```
- To see sample generated explanations:
  ```
  python explainer/sample.py --dataset {dataset}
  ```
- To evaluate generated explanations:
  ```
  python evaluation/main.py --dataset {dataset}
  ```

Supported datasets:  `amazon`, `google`, `yelp`

## Example

Below is a example of generating explanation for a specific user-item recommendation using the ``Yelp`` dataset.

### Input
- Item profile processed by GPT:
  ```
  MD Oriental Market, is summarized to attract Fans of Asian cuisine, individuals looking for a variety of Asian products, and those seeking unique and ethnic food items would enjoy MD Oriental Market. Customers interested in a well-organized, spacious, and clean grocery store with a diverse selection of Asian ingredients and products would also appreciate this location.
  ```
- User profile processed by GPT:
  ```
  This user is likely to enjoy casual American comfort food, barbecue with various meat options and tasty sauces, high-quality dining experiences with tasting menus, and authentic Italian food and beverages in cozy atmospheres.
  ```
- User/Item interaction history processed by Graph Neural Networks (GNNs)
  
### Output
- Explanation for the user-item recommendation:
  ```
  The user would enjoy this business for its vast selection of Asian ingredients, including fresh produce, sauces, condiments, and spices, making it a go-to for authentic and diverse cooking options.
  ```

## Code Structure

```
├── README.md
├── data (amazon/google/yelp)
│   ├── data.json                         # user/item profile with explanation
│   ├── trn/val/tst.pkl                   # separation of data.json
│   ├── total_trn/val/tst.csv             # user-item interactions
│   ├── user/item_emb.pkl                 # user/item embeddings
│   ├── user/item_converter.pkl           # MoE adapter
│   ├── tst_pred.pkl                      # generated explanation
│   └── tst_ref.pkl                       # ground truth explanation
├── encoder
│   ├── models                            # graph neural network (GNN) structure
│   ├── utils
│   └── train_encoder.py                  # derive user/item embeddings
├── explainer
│   ├── models
│   │   ├── explainer.py                  # XRec model
│   │   └── modeling_explainer.py         # modified PyTorch LLaMA model
│   ├── utils
│   ├── main.py                           # employ XRec                
│   └── sample.py                         # see samples of generated explanations
├── generation
│   ├── instructions                      # system prompts for user/item profile and
│   ├── explanations
│   ├── item_profile                      # generate item profile
│   │   ├── item_prompts.json
│   │   ├── item_system_prompt.json
│   │   └── generate_profile.py
│   ├── user_profile                      # generate user profile
│   │   ├── user_prompts.json
│   │   ├── user_system_prompt.json
│   │   └── generate_profile.py
│   └── explanation                       # generate ground truth explanation
│       ├── exp_prompts.json
│       ├── exp_system_prompts.json    
│       └── generate_exp.py
└── evaluation
    ├── main.py
    ├── metrics.py               
    └── system_prompt.txt                  # system prompt for GPTScore
```

## Citation
If you find XRec helpful to your research or applications, please kindly cite:
```bibtex
@article{ma2024xrec,
  title={XRec: Large Language Models for Explainable Recommendation},
  author={Ma, Qiyao and Ren, Xubin and Huang, Chao},
  journal={arXiv preprint arXiv:2406.02377},
  year={2024}
}
```
