import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer
from models.modeling_explainer import LlamaForCausalLM


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)
    

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps=8, layers=[64, 4096], dropout=0.2, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class Explainer(torch.nn.Module):
    def __init__(self, token_size=4096, user_embed_size=64, item_embed_size=64):
        super(Explainer, self).__init__()
        from huggingface_hub import login
        # Login to Hugging Face Hub
        login() 
        
        # Load the model and tokenizer
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

        # add special tokens for user and item embeddings
        special_tokens_dict = {"additional_special_tokens": ["<USER_EMBED>", "<ITEM_EMBED>", "<EXPLAIN_POS>"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        self.tokenizer.pad_token = "<pad>"
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # freeze parameters in llama
        for param in self.model.parameters():
            param.requires_grad = False

        self.user_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[user_embed_size, token_size], dropout=0.2, noise=True)
        self.item_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[item_embed_size, token_size], dropout=0.2, noise=True)

    def forward(self, user_embedding, item_embedding, input_text):
        # Convert embeddings
        converted_user_embedding = self.user_embedding_converter(user_embedding).half()
        converted_item_embedding = self.item_embedding_converter(item_embedding).half()

        # shape of tokenized_inputs['input_ids']: [batch_size, input_length]
        tokenized_inputs = self.tokenizer(
            input_text, padding=True, return_tensors="pt"
        )

        # Convert tokenized input IDs to model's embeddings
        inputs_embeds = self.model.get_input_embeddings()(tokenized_inputs['input_ids'])
        
        # Get the token ID for the <USER_EMBED> <ITEM_EMBED> token
        user_embed_token_id = self.tokenizer.convert_tokens_to_ids("<USER_EMBED>")
        item_embed_token_id = self.tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")
        explain_pos_token_id = self.tokenizer.convert_tokens_to_ids("<EXPLAIN_POS>")

        # Find the position of the <USER_EMBED> <ITEM_EMBED> <EXPLAIN_POS> token in the input embeddings
        # shape of explain_pos_position: [batch_size]
        user_embed_position = (tokenized_inputs['input_ids'] == user_embed_token_id).nonzero()[:,1:]
        item_embed_position = (tokenized_inputs['input_ids'] == item_embed_token_id).nonzero()[:,1:]
        explain_pos_position = (tokenized_inputs['input_ids'] == explain_pos_token_id).nonzero()[:,1:]

        # replace by our converted embeddings
        inputs_embeds[torch.arange(user_embed_position.shape[0]), user_embed_position[:,0], :] = converted_user_embedding
        inputs_embeds[torch.arange(item_embed_position.shape[0]), item_embed_position[:,0], :] = converted_item_embedding

        # shape of outputs.logits: [batch_size, input_length, vocab_size]
        outputs = self.model(inputs_embeds=inputs_embeds, user_embed = converted_user_embedding, item_embed = converted_item_embedding, user_embed_pos=user_embed_position, item_embed_pos=item_embed_position)
        return tokenized_inputs['input_ids'], outputs, explain_pos_position.flatten()

    def loss(self, input_ids, outputs, explain_pos_position, device):
        '''
        input_ids: [batch_size, input_length]
        outputs.logits: [batch_size, input_length, vocab_size]
        explain_pos_position: [batch_size]
        '''
        # freeze the information
        interval = torch.arange(input_ids.shape[1]).to(device)
        mask = interval[None, :] < explain_pos_position[:, None]
        input_ids[mask] = -100
        
        logits = outputs.logits
        # Shift target_ids to the right to create labels; the last token is ignored in the targets.
        shift_labels = input_ids[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = nn.CrossEntropyLoss()(shift_logits, shift_labels)
        return loss
    
    def generate(self, user_embedding, item_embedding, input_text):
        # Convert embeddings
        converted_user_embedding = self.user_embedding_converter(user_embedding).half()
        converted_item_embedding = self.item_embedding_converter(item_embedding).half()

        # shape of tokenized_inputs['input_ids']: [batch_size, input_length]
        tokenized_inputs = self.tokenizer(
            input_text, padding=True, return_tensors="pt"
        )

        # Convert tokenized input IDs to model's embeddings
        inputs_embeds = self.model.get_input_embeddings()(tokenized_inputs['input_ids'])
        
        # Get the token ID for the <USER_EMBED> <ITEM_EMBED> token
        user_embed_token_id = self.tokenizer.convert_tokens_to_ids("<USER_EMBED>")
        item_embed_token_id = self.tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")
        explain_pos_token_id = self.tokenizer.convert_tokens_to_ids("<EXPLAIN_POS>")

        # Find the position of the <USER_EMBED> <ITEM_EMBED> <EXPLAIN_POS> token in the input embeddings
        # shape of explain_pos_position: [batch_size]
        user_embed_position = (tokenized_inputs['input_ids'] == user_embed_token_id).nonzero()[:,1:]
        item_embed_position = (tokenized_inputs['input_ids'] == item_embed_token_id).nonzero()[:,1:]
        explain_pos_position = (tokenized_inputs['input_ids'] == explain_pos_token_id).nonzero()[:,1:]

        # replace by our converted embeddings
        # shape of inputs_embeds: [batch_size, input_length, hidden_size]
        inputs_embeds[torch.arange(user_embed_position.shape[0]), user_embed_position[:,0], :] = converted_user_embedding
        inputs_embeds[torch.arange(item_embed_position.shape[0]), item_embed_position[:,0], :] = converted_item_embedding

        # shape of outputs.logits: [batch_size, input_length, vocab_size]
        outputs = self.model.generate(inputs_embeds=inputs_embeds, max_new_tokens=128, user_embed = converted_user_embedding, item_embed = converted_item_embedding, user_embed_pos=user_embed_position, item_embed_pos=item_embed_position)
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text
        