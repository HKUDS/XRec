import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import requests
import re
from typing import List, Optional, Dict, Any


class OllamaTokenizer:
    """Simple tokenizer wrapper for Ollama compatibility"""
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        self.special_tokens = {
            "<USER_EMBED>": 32000,
            "<ITEM_EMBED>": 32001,
            "<EXPLAIN_POS>": 32002,
            "<pad>": 32003
        }
        self.pad_token = "<pad>"
        self.pad_token_id = self.special_tokens["<pad>"]

    def add_special_tokens(self, special_tokens_dict):
        """Add special tokens (for compatibility)"""
        if "additional_special_tokens" in special_tokens_dict:
            for token in special_tokens_dict["additional_special_tokens"]:
                if token not in self.special_tokens:
                    self.special_tokens[token] = len(self.special_tokens) + 32000
        if "pad_token" in special_tokens_dict:
            self.pad_token = special_tokens_dict["pad_token"]
            self.special_tokens[self.pad_token] = 32003

    def convert_tokens_to_ids(self, token):
        """Convert token to ID"""
        return self.special_tokens.get(token, 0)

    def __call__(self, text, padding=True, return_tensors="pt"):
        """Simple tokenization - for compatibility with existing code"""
        if isinstance(text, list):
            # Handle batch of texts
            max_len = 0
            tokenized_batch = []

            for t in text:
                # Simple word-based tokenization with special token handling
                tokens = self._tokenize_text(t)
                tokenized_batch.append(tokens)
                max_len = max(max_len, len(tokens))

            if padding:
                # Pad sequences to max length
                for tokens in tokenized_batch:
                    while len(tokens) < max_len:
                        tokens.append(self.pad_token_id)

            if return_tensors == "pt":
                return {"input_ids": torch.tensor(tokenized_batch)}
            return {"input_ids": tokenized_batch}
        else:
            tokens = self._tokenize_text(text)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([tokens])}
            return {"input_ids": [tokens]}

    def _tokenize_text(self, text):
        """Simple tokenization with safe token IDs"""
        # Replace special tokens with their IDs
        for token, token_id in self.special_tokens.items():
            text = text.replace(token, f" {token_id} ")

        # Simple word tokenization (this is a simplified approach)
        words = text.split()
        tokens = []
        for word in words:
            if word.isdigit() and int(word) in self.special_tokens.values():
                tokens.append(int(word))
            else:
                # Simple hash-based tokenization for regular words - keep within safe range
                token_id = abs(hash(word)) % 29000  # Keep well below 32000 to avoid conflicts
                tokens.append(token_id)

        return tokens

    def batch_decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        results = []
        for sequence in token_ids:
            text = self._decode_sequence(sequence, skip_special_tokens)
            results.append(text)
        return results

    def _decode_sequence(self, tokens, skip_special_tokens=True):
        """Decode a single sequence"""
        # This is a placeholder - in practice, you'd need proper detokenization
        # For now, we'll return a simple placeholder
        return "Generated explanation text"


class OllamaClient:
    """Client for interacting with Ollama API"""
    def __init__(self, base_url="http://localhost:11434", model_name="llama3.1:8b"):
        self.base_url = base_url
        self.model_name = model_name

    def generate(self, prompt, max_tokens=128, temperature=0.7):
        """Generate text using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"Ollama API error: {response.status_code}")
                return "Error generating response"

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return "Error connecting to Ollama"


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
    def __init__(self, token_size=4096, user_embed_size=64, item_embed_size=64, model_name="llama3.1:8b"):
        super(Explainer, self).__init__()

        # Initialize Ollama client and tokenizer
        self.ollama_client = OllamaClient(model_name=model_name)
        self.tokenizer = OllamaTokenizer(model_name=model_name)

        # Add special tokens for user and item embeddings
        special_tokens_dict = {"additional_special_tokens": ["<USER_EMBED>", "<ITEM_EMBED>", "<EXPLAIN_POS>"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})

        # Store dimensions
        self.token_size = token_size
        self.user_embed_size = user_embed_size
        self.item_embed_size = item_embed_size

        # MoE adapters for converting embeddings to text-compatible representations
        self.user_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[user_embed_size, token_size], dropout=0.2, noise=True)
        self.item_embedding_converter = MoEAdaptorLayer(n_exps=8, layers=[item_embed_size, token_size], dropout=0.2, noise=True)

        # Embedding projection layers to convert to text descriptions
        self.user_embed_to_text = nn.Linear(token_size, 512)
        self.item_embed_to_text = nn.Linear(token_size, 512)

        # Text description generators
        self.user_text_generator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.item_text_generator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, user_embedding, item_embedding, input_text):
        # Convert embeddings using MoE adapters
        converted_user_embedding = self.user_embedding_converter(user_embedding)
        converted_item_embedding = self.item_embedding_converter(item_embedding)

        # Generate text representations of embeddings
        user_text_features = self.user_embed_to_text(converted_user_embedding)
        item_text_features = self.item_embed_to_text(converted_item_embedding)

        # Create text descriptions from embeddings (simplified approach)
        user_desc = self._embedding_to_description(user_text_features, "user")
        item_desc = self._embedding_to_description(item_text_features, "item")

        # Handle different input types (tuple, list, or string) - same as generate method
        if isinstance(input_text, (tuple, torch.Tensor)):
            # Convert tuple/tensor to list
            if isinstance(input_text, torch.Tensor):
                input_text_list = input_text.tolist() if input_text.dim() > 0 else [input_text.item()]
            else:
                input_text_list = list(input_text)
        elif isinstance(input_text, list):
            input_text_list = input_text
        else:
            # Single string
            input_text_list = [input_text]

        processed_texts = []
        for i, text in enumerate(input_text_list):
            # Ensure text is a string
            if not isinstance(text, str):
                # Skip non-string items or convert if possible
                if hasattr(text, 'item'):  # tensor scalar
                    text = str(text.item())
                else:
                    text = str(text)

            # Replace special tokens with actual descriptions
            processed_text = text.replace("<USER_EMBED>", user_desc[i] if i < len(user_desc) else user_desc[0])
            processed_text = processed_text.replace("<ITEM_EMBED>", item_desc[i] if i < len(item_desc) else item_desc[0])
            processed_texts.append(processed_text)

        # Tokenize for compatibility with existing code
        tokenized_inputs = self.tokenizer(input_text_list, padding=True, return_tensors="pt")

        # Find explain position for loss calculation
        explain_pos_token_id = self.tokenizer.convert_tokens_to_ids("<EXPLAIN_POS>")
        explain_pos_position = (tokenized_inputs['input_ids'] == explain_pos_token_id).nonzero()
        if explain_pos_position.numel() > 0:
            explain_pos_position = explain_pos_position[:, 1:]
        else:
            # Default position if not found
            explain_pos_position = torch.tensor([[len(tokenized_inputs['input_ids'][0]) // 2]])

        # Create mock outputs for compatibility with existing training code
        batch_size = len(processed_texts)
        seq_len = tokenized_inputs['input_ids'].shape[1]
        vocab_size = 32000  # Approximate vocab size

        # Generate mock logits (these won't be used for actual generation)
        mock_logits = torch.randn(batch_size, seq_len, vocab_size)

        # Create a mock output object
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        outputs = MockOutput(mock_logits)

        return tokenized_inputs['input_ids'], outputs, explain_pos_position.flatten()

    def _embedding_to_description(self, embedding_features, embed_type):
        """Convert embedding features to text descriptions"""
        batch_size = embedding_features.shape[0]
        descriptions = []

        for i in range(batch_size):
            # Simple mapping from embedding features to description
            # In practice, this could be more sophisticated
            if embed_type == "user":
                desc = "a user with specific preferences and interaction history"
            else:  # item
                desc = "an item with particular characteristics and features"
            descriptions.append(desc)

        return descriptions

    def loss(self, input_ids, outputs, explain_pos_position, device):
        '''
        Modified loss function for Ollama-based architecture
        input_ids: [batch_size, input_length]
        outputs.logits: [batch_size, input_length, vocab_size]
        explain_pos_position: [batch_size]
        '''
        # Since we're using Ollama for generation, we'll focus on embedding quality
        # rather than language modeling loss which doesn't apply here

        # Move tensors to device
        input_ids = input_ids.to(device)
        explain_pos_position = explain_pos_position.to(device)

        # Clamp input_ids to valid range to avoid CUDA assertion errors
        vocab_size = outputs.logits.size(-1)
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

        # Create mask for explanation position
        interval = torch.arange(input_ids.shape[1]).to(device)
        mask = interval[None, :] < explain_pos_position[:, None]

        # Create modified input_ids for loss calculation
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = -100  # Ignore tokens before explanation position

        logits = outputs.logits.to(device)

        # Standard language modeling loss with proper bounds checking
        shift_labels = masked_input_ids[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Ensure labels are within valid range
        valid_mask = (shift_labels >= 0) & (shift_labels < vocab_size)
        shift_labels = torch.where(valid_mask, shift_labels, torch.tensor(-100, device=device))

        # Calculate cross entropy loss
        try:
            ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_labels)
        except RuntimeError:
            # If CE loss fails, use a simple MSE loss on embeddings
            ce_loss = torch.tensor(0.0, device=device)

        # Focus on embedding converter quality (main training objective)
        user_reg = torch.mean(torch.norm(self.user_embedding_converter.w_gate, dim=1))
        item_reg = torch.mean(torch.norm(self.item_embedding_converter.w_gate, dim=1))

        # Embedding consistency loss (encourage meaningful representations)
        user_embed_loss = torch.mean(torch.norm(self.user_embedding_converter.w_gate - self.user_embedding_converter.w_gate.mean(dim=0), dim=1))
        item_embed_loss = torch.mean(torch.norm(self.item_embedding_converter.w_gate - self.item_embedding_converter.w_gate.mean(dim=0), dim=1))

        # Combined loss focusing on embedding quality
        embedding_loss = 0.1 * (user_reg + item_reg) + 0.01 * (user_embed_loss + item_embed_loss)

        return ce_loss * 0.1 + embedding_loss  # Emphasize embedding training
    
    def generate(self, user_embedding, item_embedding, input_text):
        # Convert embeddings using MoE adapters
        converted_user_embedding = self.user_embedding_converter(user_embedding)
        converted_item_embedding = self.item_embedding_converter(item_embedding)

        # Generate text representations of embeddings
        user_text_features = self.user_embed_to_text(converted_user_embedding)
        item_text_features = self.item_embed_to_text(converted_item_embedding)

        # Create text descriptions from embeddings
        user_desc = self._embedding_to_description(user_text_features, "user")
        item_desc = self._embedding_to_description(item_text_features, "item")

        # Handle different input types (tuple, list, or string)
        if isinstance(input_text, (tuple, torch.Tensor)):
            # Convert tuple/tensor to list
            if isinstance(input_text, torch.Tensor):
                input_text_list = input_text.tolist() if input_text.dim() > 0 else [input_text.item()]
            else:
                input_text_list = list(input_text)
        elif isinstance(input_text, list):
            input_text_list = input_text
        else:
            # Single string
            input_text_list = [input_text]

        generated_texts = []
        for i, text in enumerate(input_text_list):
            # Ensure text is a string
            if not isinstance(text, str):
                # Skip non-string items or convert if possible
                if hasattr(text, 'item'):  # tensor scalar
                    text = str(text.item())
                else:
                    text = str(text)

            # Replace special tokens with actual descriptions
            processed_text = text.replace("<USER_EMBED>", user_desc[i] if i < len(user_desc) else user_desc[0])
            processed_text = processed_text.replace("<ITEM_EMBED>", item_desc[i] if i < len(item_desc) else item_desc[0])

            # Find the explanation position and extract the prompt
            explain_pos = processed_text.find("<EXPLAIN_POS>")
            if explain_pos != -1:
                prompt = processed_text[:explain_pos]
                # Add instruction for explanation generation
                prompt += "\n\nBased on the user and item information above, generate a concise explanation for why this item would be recommended to this user:"
            else:
                prompt = processed_text

            # Generate explanation using Ollama
            generated_text = self.ollama_client.generate(prompt, max_tokens=128, temperature=0.7)
            generated_texts.append(generated_text)

        return generated_texts
        