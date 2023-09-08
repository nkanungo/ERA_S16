# ERA_Speeding_up_Transformers_16
# Enhancing Transformer Efficiency
This project aims to enhance the efficiency of Transformers specifically for machine translation tasks. This will be achieved by implementing optimizations in both the model architecture and the dataset, with the goal of reducing computational overhead and improving overall efficiency.

## Model Changes

### a) Parameter Sharing for Encoder and Decoder Blocks

In this project, we've introduced parameter sharing between the encoder and decoder blocks, resulting in a reduction in the total number of parameters used.

### b) Attention Matrix Size Optimization

We dynamically adapt the size of the attention matrix according to the batch, effectively reducing computational requirements. This adjustment has led to notable speed enhancements in our Transformer model.

### c) Feed-Forward Network Dimension Reduction

We've fine-tuned the dimensions of the feed-forward network to boost the efficiency of our Transformer model, all the while preserving its performance.

## Dataset Changes

### a) Filtering Source Sentences

To streamline the dataset and expedite the training process, we apply a filter to include only source sentences with a length of fewer than 150 characters.

### b) Filtering Destination Sentences Based on Source Length



To further enhance dataset optimization, we employ a filtering mechanism for destination sentences based on the length of their corresponding source sentences.

### c) Custom Collate Function

To facilitate efficient data batching and preprocessing for training, we have introduced a custom collate function named collate_b. This function is specifically designed to seamlessly integrate with PyTorch's DataLoader. Below, we provide a detailed breakdown of its functionality:



def collate_b(pad_token, b):
    # Determine the maximum batch sequence length for encoder and decoder inputs
    e_i_batch_length = list(map(lambda x: x['encoder_input'].size(0), b))
    d_i_batch_length = list(map(lambda x: x['decoder_input'].size(0), b))
    seq_len = max(e_i_batch_length + d_i_batch_length)
    
    # Initialize lists to store source and target text data
    src_text = []
    tgt_text = []
    
    # Loop through each item in the batch
    for item in b:
        # Pad encoder input, decoder input, and labels to match the maximum sequence length
        item['encoder_input'] = torch.cat([item['encoder_input'],
                                           torch.tensor([pad_token] * (seq_len - item['encoder_input'].size(0)), dtype=torch.int64)], dim=0)
        item['decoder_input'] = torch.cat([item['decoder_input'],
                                           torch.tensor([pad_token] * (seq_len - item['decoder_input'].size(0)), dtype=torch.int64)], dim=0)
        item['label'] = torch.cat([item['label'],
                                   torch.tensor([pad_token] * (seq_len - item['label'].size(0)), dtype=torch.int64)], dim=0)
        
        # Collect source and target text for each item
        src_text.append(item['src_text'])
        tgt_text.append(item['tgt_text'])
    
    # Return a dictionary containing the processed data
    return {
        'encoder_input': torch.stack([o['encoder_input'] for o in b]),  # (bs, seq_len)
        'decoder_input': torch.stack([o['decoder_input'] for o in b]),  # (bs, seq_len)
        'label': torch.stack([o['label'] for o in b]),  # (bs, seq_len)
        "encoder_mask": torch.stack([(o['encoder_input'] != pad_token).unsqueeze(0).unsqueeze(1).int() for o in b]),  # (bs, 1, 1, seq_len)
        "decoder_mask": torch.stack([(o['decoder_input'] != pad_token).int() & causal_mask(o['decoder_input'].size(dim=-1)) for o in b]),
        "src_text": src_text,
        "tgt_text": tgt_text
    }



This custom collate function efficiently handles padding, masking, and text collection for the batched data, ensuring that it's ready for training with the Transformer model.

# In this function:

To guarantee uniform padding, we determine the maximum sequence length within the data batch.
The input sequences (encoder_input, decoder_input, and label) are padded using the pad_token to align with the maximum sequence length.
Masks (encoder_mask and decoder_mask) are generated to manage padding elements during model training.
Source and target texts are collected for future reference.
Ultimately, the function yields a dictionary containing batched tensors and masks, making it well-suited for integration with the Transformer model. This tailored collate function streamlines data preprocessing and enhances batch-handling efficiency during training.



##  Usage

To run and further explore:
```bash
python main.py
```









