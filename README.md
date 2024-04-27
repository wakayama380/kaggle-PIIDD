<h1>5th Place Solution</h1>

<h2>Introduction</h2>
<p>First and foremost, I would like to express my gratitude to the host and Kaggle staff for hosting this competition. Also, I would like to thank the Kaggle community for sharing various datasets and insights. Especially, I am grateful to @nbroad, @pjmathematician, @valentinwerner, and @mpware as their datasets and notebooks greatly contributed to our solution. Thank you so much! And, I am grateful to my teammates, @takai380 and @minfuka. (Congratulations to @takai380 on competition master title!)</p>

<h2>Ensemble Part</h2>
<table>
  <tr>
    <th>Model</th>
    <th>Max Length (Train/Inference)</th>
    <th>Weight</th>
    <th>CV</th>
    <th>Public</th>
    <th>Private</th>
  </tr>
  <tr>
    <td>Ryota (ryotak12)</td>
    <td>128/128</td>
    <td>10</td>
    <td>0.979</td>
    <td>0.973</td>
    <td>0.960</td>
  </tr>
  <tr>
    <td>Takai (takai380)</td>
    <td>512/512</td>
    <td>13</td>
    <td>0.977</td>
    <td>0.975</td>
    <td>0.965</td>
  </tr>
  <tr>
    <td>Minfuka (minfuka)</td>
    <td>1536/4096</td>
    <td>8</td>
    <td>None</td>
    <td>0.972</td>
    <td>0.967</td>
  </tr>
</table>

<h2>Ryota Part</h2>
<p>I created a total of 6 models for the final ensemble.</p>
<p>Common for all models: backbone: deberta-v3-large, task: token classification, full data training, external dataset, max_length: 128, train_overlap=96, eval_overlap=64, positional features, EMA, layer freezing. Freezing the backbone for the first epoch led to faster convergence. From the second epoch, no layers were frozen. Loss weight: "O": 1, Other: 10, optimizer: AdamW, scheduler: Cosine Annealing.</p>
<p>Model Specifics: with/without prefix. Four models were trained using labels with prefixes, while two models were trained using labels without prefixes.</p>
<p>Additional Training: I basically trained the models using the nbroad 's dataset as additional data. However, for some models, I first trained models for a few epochs using the competition data combined with datasets from nbroad, mpware, and pjmathematician. Then, I retrained these models for a few more epochs using only the competition data and nbroad 's dataset.</p>
<p>Max_length: I basically trained the models with a max_length of 128, but for the sake of diversity, I trained one model with a max_length of 512.</p>
<p>Post Process: Before ensemble, I applied the following post-processing steps to each single model: Removed whitespace, ensured prefix consistency, removed false positives using regular expressions tailored to each PII type.</p>

<h2>Takai380 Part</h2>
<p>Model1: deberta-v3-large (cv/public/private → 0.974/0.970/0.961), max_len=512, stride=128. Datasets: @mpware, @pjmathematician, @nbroad. Weight of non-'O' losses increased by 5 times.</p>
<p>Model2: deberta-v3-large+lstm (cv/public/private → 0.965/0.972/0.955), max_len=512, stride=128. Dataset: nbroad.</p>
<p>Model3: deberta-v3-large (cv/public/private → 0.973/0.967/0.969), max_len=512, stride=128. Dataset: nbroad. Weight of non-'O' losses increased by 5 times.</p>
<p>Model4: deberta-v3-large (cv/public/private → 0.970/?/?), max_len=1024, stride=256. Dataset: nbroad. Add lstm layer following deberta-v3-large. Weight of non-'O' losses increased by 5 times. (This model was not used in the selected private best model.)</p>
<p>Common for all models: B- and I- tags are not used. Basic rule-based processing: NAME_STUDENT: Rejects if not starting with a capital letter followed by lowercase letters. If one character, remove. URL and EMAIL formats must be adhered to, or they are rejected. Strict rule-based processing: If an article precedes the prediction of NAME_STUDENT, it is removed unless "'s" follows, in which case it is not removed. Examples: - Articles ('The', ' the', 'a', 'and') often precede non-name terms. If 'at' is before, it's likely a location and is excluded. However, there are exceptions when followed by "'s" as it may not fit the rules as intended: ~ the Takai's house ~. In such cases, the rule is not applied. This is a personal observation, but I am surprised to see that the scores (0.969) achieved with a single model, either similar to or higher than those obtained by ensemble methods combining CV and public LB scores. Does this suggest that the labels already detectable by a single model are achievable, while others pose more difficulty for machine learning models to detect?</p>

<h2>Minfuka Part</h2>
<p>Common for all models: Base-Model: deberta-v3-large (Freezing embeddings, Freezing first 6 layers), max_len=1536. Datasets: training_data: competition training data (4/5) + external data, validation_data: competition training data (1/5). TrainingArguments: fp16=True, learning_rate=2e-5, num_train_epochs=3, per_device_train_batch_size=8, gradient_accumulation_steps=2, gradient_checkpointing=True, logging_steps=50, evaluation_strategy='steps', eval_steps=50, save_strategy="steps", save_steps=50, load_best_model_at_end=True, lr_scheduler_type='cosine', metric_for_best_model='f5', greater_is_better=True, warmup_ratio=0.1, loss_function: CrossEntropyLoss(weight=class_weight) class_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1] ← Only Label 'O' is 0.1. External data (Thanks for all external-data author): I use various datasets to generate diversity. Model1 using @mpware 's dataset (https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays). Model2 using dataset in @valentinwerner 's notebook (https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset/output). Model3 using "moredata" in @valentinwerner 's notebook (https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset/output). Rule-based processing: Nothing. Inference: Notebooks I referred to: https://www.kaggle.com/code/valentinwerner/945-deberta-3-base-striding-inference (Special thanks @valentinwerner). Difference from training: INFERENCE_MAX_LENGTH=4096, STRIDE=384. Ensemble: ensemble (use models mentioned above: Model1, Model2, and Model3.) ensemble_pred = (Model1 + Model2 + Model3) / 3 ← before softmax. Above inference score is Public:0.972 Private:0.967. (Add takai-rule-based processing, score is Public:0.971 Private:0.968) URL and EMAIL formats must be adhered to, or they are rejected. Examples: - Articles ('The', ' the', 'a', 'and') often precede non-name terms. If 'at' is before, it's likely a location and is excluded.</p>
