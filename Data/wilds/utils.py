from wilds.datasets.wilds_dataset import WILDSSubset, WILDSDataset
import torch
import pandas as pd
import numpy as np
from wilds import get_dataset
from transformers import BertTokenizer, BertForSequenceClassification

def camelyon_preprocess(dataset_camelyon17: WILDSDataset, min_domain_label_cnt: int = 100) -> WILDSDataset:
        print(f"Creating data with {min_domain_label_cnt} min_domain_label_cnt")
        camelyon_domains = dataset_camelyon17._metadata_df['patient'].values.astype('int')
        dataset_camelyon17._metadata_array = torch.Tensor(camelyon_domains)
        camelyon_indices = (
            dataset_camelyon17
            ._metadata_df
            .assign(
                domain_label_count = lambda df: df.groupby(['patient', 'tumor']).transform('size')
            )
            .loc[lambda df: df['domain_label_count']>min_domain_label_cnt] # 200
            .index
            .values
        )
        dataset_camelyon17_subset = WILDSSubset(
            dataset_camelyon17,
            indices=camelyon_indices,
            transform = lambda image: torch.tensor(np.array(image)).permute(2, 0, 1),
        )
        return dataset_camelyon17_subset


FMOW_FILTERED_LABELS = [42, 11, 36]
FMOW_LABELS_MAP = {label:i for i,label in enumerate(FMOW_FILTERED_LABELS)}
def fmow_preprocess(dataset_fmow: WILDSDataset, min_domain_label_cnt: int = 100) -> WILDSDataset:
        print(f"Creating data with {min_domain_label_cnt} min_domain_label_cnt")
        metadata_df=pd.DataFrame(
            dataset_fmow._metadata_array.numpy(), 
            columns = dataset_fmow._metadata_fields
        )
        i = 0
        d = {}
        for reg in metadata_df.region.unique():
            for year in metadata_df.year.unique():
                d[reg,year]= i
                i+=1
        fmow_domains = np.array([d[reg, year] for reg, year in metadata_df[['region', 'year']].values])
        dataset_fmow._metadata_array = torch.Tensor(fmow_domains)
        print(f"modifying labels according to the map: {FMOW_LABELS_MAP}")
        dataset_fmow._y_array = torch.tensor([
            FMOW_LABELS_MAP[y.item()] if y in FMOW_FILTERED_LABELS else -1 
            for y in dataset_fmow._y_array
        ])

        fmow_indices = (
            metadata_df
            .assign(domain = fmow_domains)
            .loc[lambda df: df['y'].isin(FMOW_FILTERED_LABELS)]
            .assign(
                domain_label_count = lambda df: df.groupby(['domain', 'y'])['year'].transform('size')
            )
            .loc[lambda df: df['domain_label_count']>min_domain_label_cnt] # 200
            .index
            .values
        )

        dataset_fmow_subset = WILDSSubset(
            dataset_fmow,
            indices=fmow_indices,
            transform = lambda image: torch.tensor(np.array(image)).permute(2, 0, 1),
        )

        assertion_cond = all(
            torch.isin(
                dataset_fmow_subset.y_array, 
                torch.tensor(np.arange(len(FMOW_FILTERED_LABELS)))
            )
        )
        assert assertion_cond, "dataset labels are not valid, perhaps something is wrong with the labels modification"
        return dataset_fmow_subset


def amazon_preprocess(dataset_amazon: WILDSDataset, min_domain_label_cnt: int = 100) -> WILDSDataset:
    amazon_df = pd.DataFrame(
        dataset_amazon.metadata_array, 
        columns= dataset_amazon._metadata_fields
    )

    amazon_domains = amazon_df['user'].values.astype('int')
    dataset_amazon._metadata_array = torch.Tensor(amazon_domains)

    amazon_indices = (
        amazon_df
        .assign(domain = amazon_domains)
        .assign(
            domain_label_count = lambda df: df.groupby(['domain', 'y']).transform('size')
        )
        .loc[lambda df: df['domain_label_count']>min_domain_label_cnt]
        .index
        .values
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    embedding_layer = model.bert.embeddings.word_embeddings
    def transform(X):
        tokenizer_oupout =  tokenizer(
            X,
            padding = 'max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        embeddings = embedding_layer(tokenizer_oupout['input_ids'])[:, 1:-1, :]
        mask = tokenizer_oupout['attention_mask'][:, 1:-1].unsqueeze(-1)
        avg_embedding = ((embeddings * mask).sum(dim=1) / mask.sum(dim=1)).squeeze(0)
        return avg_embedding.detach()
    
    
    dataset_amazon_subset = WILDSSubset(
        dataset_amazon,
        indices=amazon_indices,
        transform = transform,
    )

    return dataset_amazon_subset




IWILDCAM_FILTERED_LABELS = [0, 147, 146]
IWILDCAM_LABELS_MAP = {label:i for i,label in enumerate(IWILDCAM_FILTERED_LABELS)}
def iwildcam_preprocess(dataset_iwildcam: WILDSDataset, min_domain_label_cnt: int = 100) -> WILDSDataset:
    print(f"Creating data with {min_domain_label_cnt} min_domain_label_cnt")
    ## Create df of metadata
    iwildcam_metadata_df = (
          pd.DataFrame(
            dataset_iwildcam.metadata_array, 
            columns= dataset_iwildcam._metadata_fields
        )
    )
    ##Choose indices to filter by size of domain X label
    iwildcam_indices = (
        iwildcam_metadata_df.loc[lambda df: df['y'].isin(IWILDCAM_FILTERED_LABELS)]
        .assign(
            domain_label_count = lambda df: df.groupby(['location', 'y']).transform('size')
        )
        .loc[lambda df: df['domain_label_count']>min_domain_label_cnt]
        .index
        .values
    )
    
    #Change metadata_array to be the domains
    iwildcam_domains = iwildcam_metadata_df['location'].values.astype('int')
    dataset_iwildcam._metadata_array = torch.Tensor(iwildcam_domains)

    print(f"modifying labels according to the map: {FMOW_LABELS_MAP}")
    dataset_iwildcam._y_array = torch.tensor([
        IWILDCAM_LABELS_MAP[y.item()] if y in IWILDCAM_FILTERED_LABELS else -1 
        for y in dataset_iwildcam._y_array
    ])

    #Create the filtered subset
    img_height, img_width, _ = np.array(dataset_iwildcam[0][0]).shape
    dataset_iwildcam_filtered = WILDSSubset(
        dataset_iwildcam,
        indices=iwildcam_indices,
        transform = lambda image: (
            torch.tensor(
                np.array(image.crop((img_width*0.05, img_height*0.05, img_width*0.95, img_height*0.95)))
            )
            .permute(2, 0, 1)
        ),
    )
    assertion_cond = all(
        torch.isin(
              dataset_iwildcam_filtered.y_array, 
              torch.tensor(np.arange(len(IWILDCAM_FILTERED_LABELS)))
        )
    )
    assert assertion_cond, "dataset labels are not valid, perhaps something is wrong with the labels modification"

    return dataset_iwildcam_filtered
