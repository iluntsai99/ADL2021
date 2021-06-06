# ADL HW3 @NTU, 2021 spring

B06902135 資工四 蔡宜倫

## 1. Model (2%)

+ **Model (1%)**

    + **Describe the model architecture and how it works on text summarization.**

        + **Configuration:**

            ```json
            {
              "_name_or_path": "google/mt5-small",
              "architectures": [
                "MT5ForConditionalGeneration"
              ],
              "d_ff": 1024,
              "d_kv": 64,
              "d_model": 512,
              "decoder_start_token_id": 0,
              "dropout_rate": 0.1,
              "eos_token_id": 1,
              "feed_forward_proj": "gated-gelu",
              "initializer_factor": 1.0,
              "is_encoder_decoder": true,
              "layer_norm_epsilon": 1e-06,
              "model_type": "mt5",
              "num_decoder_layers": 8,
              "num_heads": 6,
              "num_layers": 8,
              "pad_token_id": 0,
              "relative_attention_num_buckets": 32,
              "tie_word_embeddings": false,
              "tokenizer_class": "T5Tokenizer",
              "transformers_version": "4.5.0",
              "use_cache": true,
              "vocab_size": 250112
            }
            ```

+ **Preprocessing (1%)**

    + **Describe your preprocessing (e.g. tokenization, data cleaning and etc.)**
    + 

## 2. Training (2%)

+ **Hyperparameter (1%)**

    + **Describe your hyperparameter you use and how you decide it.**

+ **Learning Curves (1%)**

    + **Plot the learning curves (ROUGE versus training steps)**

        <img src="/Users/iluntsai99/Desktop/learningCurve.png" alt="learningCurve" style="zoom:72%;" />

## 3. Curves (1%)

+ **Stratgies (2%)**
    + **Describe the detail of the following generation strategies:**
        + **Greedy**
        + **Beam Search**
        + **Top-k Sampling**
        + **Top-p Sampling**
        + **Temperature**
+ **Hyperparameters (4%)**
    + **Try at least 2 settings of each strategies and compare the result.** 
    + **What is your final generation strategy? (you can combine any of them)**

## 4. Bonus: Applied RL on Summarization (2%)

+ **Algorithm (1%)**
+ **Describe your RL algorithms, reward function, and hyperparameters.**
+ **Compare to Supervised Learning (1%)**
+ **Observe the loss, ROUGE score and output texts, what differences can you find?**
