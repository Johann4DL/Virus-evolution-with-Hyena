from hyena_simp import Config, HyenaConfig

hyena_config = HyenaConfig(
  embedding_dim = 1000,
  d_model= 1, #386,
  n_layers=2,
  vocab_size=13, #len(vocabulary),
  d_embed=5,
  d_filter_mlp=64,
  n_filter_layers=4,
  context_length=29660, #CONTEXT_LENGTH,
  short_conv_size=3,
  order=2,
  pdrop_hyena=0.0,
  pdrop_embed=0.2,
  omega=12,
  epochs=200,
  learning_rate=6e-4,
  betas=(0.9, 0.98),
  weight_decay=1,
  device_type="gpu",  # cpu, gpu
  precision="bf16",  # 32, 16, 16-mixed, bf16
  batch_size=2,
  num_workers=4,
)
