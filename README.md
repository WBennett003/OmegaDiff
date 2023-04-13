# OmegaDiff

This project trains OmegaPLM layers and wraps them with Adapative Layer Norms with conditioning data to learn
to denoise embedded protein sequences (DDPM). The model is conditioned with the degree of noise added t, the
reaction hash - create with digital reaction fingerprint (DRFP) - and a sequence mask to indicate what AA to 
denoise. 
