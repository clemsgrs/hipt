# hipt

TODO:
- [ ] change `nn.Parameter` instantiation from `nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))` to `nn.Parameter(nn.init.trunc_normal_(torch.zeros(1, num_patches + 1, embed_dim), mean=0.0, std=0.02))`
- [ ] when switching `img_size` argument from `[224]` to `[256]`, make sure the positional enmedding is trainable!