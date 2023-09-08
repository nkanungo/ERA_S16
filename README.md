# ERA_Speeding_up_Transformers_16
# Speeding up Transformers

This project focuses on optimizing Transformers for machine translation by making both model and dataset changes to reduce computation time and improve efficiency.

## Model Changes

### a) Parameter Sharing for Encoder and Decoder Blocks

In this project, we have implemented parameter sharing between encoder and decoder blocks, reducing the overall number of parameters.

### b) Attention Matrix Size Optimization

We dynamically adjust the attention matrix size as per the batch to reduce computations, leading to significant speed improvements in our transformer model.

### c) Feed-Forward Network Dimension Reduction

We have optimized the feed-forward network dimensions to enhance the efficiency of our Transformer model while maintaining performance.

## Dataset Changes

### a) Filtering Source Sentences

We filter source sentences with a length of less than 150 characters to streamline the dataset and speed up training.

### b) Filtering Destination Sentences Based on Source Length

To further optimize the dataset, we filter destination sentences based on the length of the corresponding source sentences.

### c) Custom Collate Function

To efficiently batch and preprocess data for training, we've implemented a custom collate function named `collate_b`. This function is designed to work seamlessly with PyTorch's DataLoader. Below is a breakdown of its functionality:

            ```python
            def collate_b(pad_token, b):
                e_i_batch_length = list(map(lambda x: x['encoder_input'].size(0), b))
                d_i_batch_length = list(map(lambda x: x['decoder_input'].size(0), b))
                seq_len = max(e_i_batch_length + d_i_batch_length)
                src_text = []
                tgt_text = []
                for item in b:
                    item['encoder_input'] = torch.cat([item['encoder_input'],
                                                       torch.tensor([pad_token] * (seq_len - item['encoder_input'].size(0)), dtype=torch.int64), ], dim=0)
                    item['decoder_input'] = torch.cat([item['decoder_input'],
                                                       torch.tensor([pad_token] * (seq_len - item['decoder_input'].size(0)), dtype=torch.int64), ], dim=0)
            
                    item['label'] = torch.cat([item['label'],
                                               torch.tensor([pad_token] * (seq_len - item['label'].size(0)), dtype=torch.int64), ], dim=0)
            
                    src_text.append(item['src_text'])
                    tgt_text.append(item['tgt_text'])
                return {
                    'encoder_input': torch.stack([o['encoder_input'] for o in b]),  # (bs, seq_len)
                    'decoder_input': torch.stack([o['decoder_input'] for o in b]),  # bs, seq_len)
                    'label': torch.stack([o['label'] for o in b]),  # (bs, seq_len)
                    "encoder_mask": torch.stack([(o['encoder_input'] != pad_token).unsqueeze(0).unsqueeze(1).int() for o in b]),  # (bs,1,1,seq_len)
                    "decoder_mask": torch.stack([(o['decoder_input'] != pad_token).int() & causal_mask(o['decoder_input'].size(dim=-1)) for o in b]),
                    "src_text": src_text,
                    "tgt_text": tgt_text
                }
                 ```


In this function:

- We compute the maximum sequence length within the batch of data to ensure consistent padding.
- We pad the input sequences (`encoder_input`, `decoder_input`, and `label`) with the `pad_token` to match the maximum sequence length.
- We create masks (`encoder_mask` and `decoder_mask`) to handle padding elements during model training.
- We gather source and target texts for reference.
- Finally, the function returns a dictionary containing batched tensors and masks, making it ready for use with the Transformer model. This custom collate function helps optimize data preprocessing and ensures efficient batch handling during training.



##  Usage

To run and further explore:
```bash
python main.py
```

##  License
I have used Apache 2.0 for this task. 

## Acknowledgments
I would  like to acknowledge Rohan Shravan and students of the school of AI for their valuable guidance.








