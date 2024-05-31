# Fake News Classification using Machine Learning Models

## Project Overview

In today's digital age, the proliferation of fake news poses a significant challenge, undermining trust in institutions and contributing to societal unrest. Leveraging machine learning advancements, this project employs Natural Language Processing techniques to swiftly and accurately classify news articles as real or fake. Specifically, it explores the efficacy of transformer-based encoder models like DistilBERT and ELECTRA due to their computational efficiency. Furthermore, to enhance model interpretability, explainable AI tools such as BertViz are utilized. The project evaluates model performance using metrics like accuracy, precision, recall, F1-score, and constructs confusion matrices to assess effectiveness.

## Data

 [WELFake dataset](https://zenodo.org/records/4561253) sourced from a merger of four prominent news datasets was used. It comprises a diverse collection of news articles, encompassing both real and fake content. With 72,134 entries, it offers a well-balanced and comprehensive resource for training and evaluating fake news detection models. Featuring columns for unique identifiers, titles, full text, and binary labels indicating real or fake, this dataset facilitates the development of robust machine learning algorithms. Its documentation and ease of use further enhance its suitability for research and experimentation in the realm of fake news detection.

 License:

 ## Feature Engineering
- Null Values and Duplicates Handling: Null values and duplicate entries were removed from the dataset to ensure data cleanliness.
- Text Preprocessing: The "text" and "title" columns were merged into a unified "text" column. Preprocessing steps included lowercasing, punctuation removal, stop-word removal, URL removal, and lemmatization to standardize text data.
- Dataset Sampling: Given the large dataset size of 71,537 samples, a judicious sampling strategy was implemented to balance computational resources with data representativeness. This approach involved extracting a subset of the original dataset for more manageable computational requirements while preserving data diversity and volume.
- Data Distribution Visualization: The distribution of labels (0 for fake news, 1 for real news) across training, validation, and test sets was visualized to assess model bias and generalizability. The balanced distribution between classes in all three data splits facilitates effective model training by mitigating bias and enhancing generalizability to unseen data.

 <p align="center">
    <img src="https://github.com/amruthapurnavadrevu/Fake-News-Classification/blob/main/Visualisations/DatasetSampling.png" alt="Data Distribution After Sampling" width="350"/>
</p>

## Model Building

- Model Selection: DistilBERT and ELECTRA architectures chosen for fake news classification.

- Pre-processing Pipeline:
Tokenization using AutoTokenizer and ElectraTokenizer for DistilBERT and ELECTRA, respectively.
Pre-process function for tokenization and sequence length standardization.
Data collator for organizing pre-processed data into batches.

- Fine-tuning Process:
Freezing bottom layers to leverage pre-trained knowledge and mitigate overfitting.
Hyperparameter tuning for efficient model convergence, exploring learning rates and training epochs.

- Preventing Overfitting:
Early stopping halts training when validation performance plateaus.
Weight decay regularization penalizes large parameter values during optimization.

Result: Models equipped to effectively learn from limited training data while generalizing well to unseen instances.

## Findings

- Constrained computational resources impacted model execution, utilizing only half of the dataset, reducing data volume and potentially affecting generalization.
- Limited dataset and computational constraints increased the risk of overfitting, addressed through measures like freezing bottom layers, early stopping, and weight decay regularization.
- DistilBERT showed faster convergence and training speed but slightly lower accuracy compared to ELECTRA.
- ELECTRA achieved higher accuracy with longer training times due to its enhanced model capacity and discriminator pre-training strategy.
- Analysis using confusion matrices supported these trends, with ELECTRA demonstrating superior precision in classifying real and fake news articles compared to DistilBERT.

| Model      | Accuracy | Precision | Recall  | F1-score |
|------------|----------|-----------|---------|----------|
| DistilBERT | 0.9853   | 0.9839    | 0.9866  | 0.9853   |
| ELECTRA    | 0.9907   | 0.9919    | 0.9893  | 0.9906   |

 <p>
    <img src="https://github.com/amruthapurnavadrevu/Fake-News-Classification/blob/main/Visualisations/ConfusionMatrices.png" alt="Confusion Matrices" width="500"/>
</p>

## Limitations

- Ad Hoc Dataset Creation: Datasets created during crises like COVID-19 may lack generalizability beyond those events due to their specificity to particular circumstances.
   
- Small Sizes and Imbalances: Limited dataset sizes and imbalances between fake and real news instances pose significant challenges for deep neural networks, which require extensive and balanced data for effective training.
   
- Scarcity of Labelled Data: The scarcity of large-scale labelled information presents a significant obstacle to effectively detecting fake news, particularly in rapidly evolving or emerging events where labelled data is sparse.
   
- Difficulty in Adaptation: Detection algorithms struggle to adapt to rapidly evolving situations without sufficient labelled data, limiting their ability to effectively discern fake news patterns in dynamic environments.
   
- Limited Model Explainability: There is a lack of sufficient exploration into model explainability, which hampers the transparency and trustworthiness of fake news detection models, making it challenging to understand the rationale behind model decisions.
   
- Discrepancy in Evaluation: Evaluation on fully labelled and balanced experimental datasets does not adequately reflect real-world scenarios where data is predominantly unlabelled and highly imbalanced, raising concerns about the real-world relevance and robustness of detection models.

Additionally, the availability of computational resources significantly impacts the efficacy of fake news classification models. While downsizing datasets can alleviate computational strain, it may inadvertently limit the models' exposure to diverse data patterns, potentially hindering their adaptability and generalizability to real-world news instances. Balancing computational efficiency with model generalizability remains a challenge in the realm of fake news classification.

## Conclusion

In conclusion, this research investigated the efficacy of DistilBERT and ELECTRA models for fake news classification, leveraging transfer learning to fine-tune them on a benchmark dataset. Both models achieved impressive accuracy above 98.5%, underscoring the value of transfer learning. However, challenges such as limited and skewed training data, along with computational resource constraints, were identified as significant factors influencing model performance. Acknowledging these limitations, further research exploring different data distributions and mitigation strategies like data augmentation is warranted. Overall, this study contributes to the advancement of transformer-based approaches for fake news classification, highlighting the importance of addressing data quality and computational resources for optimized model performance in combating misinformation.












