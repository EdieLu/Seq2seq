# Seq2seq: RNN-based NMT

Standard encoder-decoder NMT (following Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation Y. Wu et el)

## Prerequisites

- python 3.6
- torch 1.2
- tensorboard 1.14+
- CUDA 9

## Data

- Source / target files: one sentence per line
- Source / target vocab files: one vocab per line, the top 5 fixed to be
`<pad> <unk> <s> </s> <spc>` as defined in `utils/config.py`

## Train

To train the model - check `Examples/train.sh`

- `train_path_src` - path to source file for training
- `train_path_tgt` - path to target file for training
- `dev_path_src` - path to source file for validation (default set to `None`)
- `dev_path_tgt` - path to target file for validation (default set to `None`)
- `path_vocab_src` - path to source vocab list
- `path_vocab_tgt` - path to target vocab list
- `load_embedding_src` - load pretrained src embedding if provided
- `load_embedding_tgt` - load pretrained target embedding if provided
- `use_type` - `word` or tokenise into `char`
- `save` - dir to save the trained model
- `random_seed` - set random seed
- `share_embedder` - share embedding matrix across source and target
- `embedding_size_enc` - source embedding size
- `embedding_size_dec` - target embedding size
- `hidden_size_enc` - encoder hidden size
- `num_bilstm_enc` - number of encoder BiLSTM layers
- `num_unilstm_enc` - number of encoder UniLSTM layers (default `0`)
- `hidden_size_dec` - decoder hidden size
- `num_unilstm_dec` - number of decoder UniLSTM layers
- `att_mode` - attention mode `bahdanau | bilinear | hybrid `
- `hidden_size_att` - only used if `att_mode` is set to `hybrid`
- `residual` - residual connection across LSTM layers
- `hidden_size_shared` - transformed attention output hidden size
- `max_seq_len` - maximum sequence length, longer sentences filtered out in training
- `batch_size` - batch size
- `batch_first` - set to `True`
- `seqrev` - train seq2seq in reverse order
- `eval_with_mask` - compute loss on non `<pad>` tokens (default `True`)
- `scheduled_sampling` - scheduled sampling
- `teacher_forcing_ratio` - probability to run in teacher forcing mode, set to `1.0` for teacher forcing to be used throughout
- `dropout` - dropout rate
- `embedding_dropout` - embedding dropout rate
- `num_epochs` - number of epochs
- `use_gpu` - set to `True` if GPU device is available
- `learning_rate` - learning rate
- `max_grad_norm` - gradient clipping
- `checkpoint_every` - number of batches trained for 1 checkpoint saved (if `dev_path*` not given, save after every epoch)
- `print_every` - number of batches trained for train losses printed
- `max_count_no_improve` - used when `dev_path*` is given, number of batches  trained (with no improvement in accuracy on dev set) before roll back
- `max_count_num_rollback` - reduce learning rate if rolling back for multiple times
- `keep_num` - number of checkpoint kept in model dir (used if `dev_path*` is given)
- `normalise_loss` - normalise loss on per token basis
- `minibatch_split` - if OOM, split batch into minibatch (note gradient descent still is done per batch, not minibatch)

## Test

To test the model - check `Examples/translate.sh`

- `test_path_src` - path to source text
- `seqrev` - translate in reverse order or not
- `path_vocab_src` - be consistent with training
- `path_vocab_tgt` - be consistent with training
- `use_type` - be consistent with training
- `load` - path to model checkpoint
- `test_path_out` - path to save the translated text
- `max_seq_len` - maximum translation sequence length (set to be at least larger than the maximum source sentence length)
- `batch_size` - batch size in translation, restricted by memory
- `use_gpu` - set to `True` if GPU device is available
- `beam_width` - beam search decoding
- `eval_mode` - default `1` (other modes for debugging)
