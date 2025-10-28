import torch

# class SelfAttentionRelevance:
#     """ Chefer, H., Gur, S. & Wolf, L. Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers. 
#     Preprint at https://doi.org/10.48550/arXiv.2103.15679 (2021).
#     """

#     def __init__(self, model):
#         """
#         Initializes the SelfAttentionRelevance class with a TransformerNodeEncoder.

#         Parameters
#         ----------
#             model: torch.nn.Module
#                 The model to be used for generating relevance.
#         """
#         self.model = model

#     @staticmethod
#     def avg_heads(attn_map, grad):
#         """
#         Rule 5 from Chefer et al.: Averages the heads in the attention map after applying gradients.

#         Parameters
#         ----------
#             attn_map: Tensor
#                 Attention weight (Key and Query dependent), shape: BHxSxS
#             grad: Tensor
#                 Gradients to apply to the attention map, shape: BHxSxS

#         Returns
#         -------
#             Tensor
#                 Averaged attention map after applying gradients.
#         """
#         attn_map = attn_map.reshape(-1, attn_map.shape[-2], attn_map.shape[-1])
#         grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
#         attn_map = grad * attn_map
#         attn_map = attn_map.clamp(min=0).mean(dim=0)
#         return attn_map

#     @staticmethod
#     def apply_self_attention_rules(R_ss, cam_ss):
#         """
#         Rule 6 from Chefer et al.: Applies self-attention rules to update the relevance score.

#         Parameters
#         ----------
#             R_ss: Tensor
#                 Relevance score matrix.
#             cam_ss: Tensor
#                 Attention map.

#         Returns
#         -------
#             Tensor
#                 Updated relevance score matrix.
#         """
#         return torch.matmul(cam_ss, R_ss)

#     def generate_relevance(self, padded_h_node, src_padding_mask, category_index=None):
#         """
#         Generates the relevance score for a given input and category index.

#         Parameters
#         ----------
#             padded_h_node: Tensor
#                 Padded input node representations.
#             src_padding_mask: Tensor
#                 Padding mask for the input.
#             category_index: List[int], optional
#                 List of indices to consider in the category mask, default is None.
#         """
#         output, _ = self.model(padded_h_node, src_padding_mask, register_hook=True)
#         category_mask = torch.zeros(output.size())
#         if category_index is not None:
#             category_mask[category_index, :, :] = 1
#         loss = (output * category_mask).sum()
#         self.model.zero_grad()
#         loss.backward(retain_graph=True)

#         num_tokens = self.model.transformer_encoder.layers[0].get_attn_output().shape[0]
#         print(num_tokens)

#         #I = torch.eye(num_tokens, num_tokens).cuda()
#         I = torch.eye(num_tokens, num_tokens)

#         for idx, encoder in enumerate(self.model.transformer_encoder.layers):
#             attn_out_weights = encoder.get_attn_output_weights()
#             attn_grad = encoder.get_attn_gradients()

#             attn_map = self.avg_heads(attn_out_weights, attn_grad)
#             #I += self.apply_self_attention_rules(I.cuda(), attn_map.cuda())
#             I += self.apply_self_attention_rules(I, attn_map)
        
#         return I


class SelfAttentionRelevance:
    """ Chefer, H., Gur, S. & Wolf, L. Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers. 
    Preprint at https://doi.org/10.48550/arXiv.2103.15679 (2021).
    """

    def __init__(self, transformer_encoder):
        """
        Initializes the SelfAttentionRelevance class with a TransformerNodeEncoder.

        Parameters
        ----------
            transformer_encoder: torch.nn.Module
                The transformer encoder to be used for generating relevance.
        """
        self.transformer_encoder = transformer_encoder

    @staticmethod
    def avg_heads(attn_map, grad):
        """
        Rule 5 from Chefer et al.: Averages the heads in the attention map after applying gradients.

        Parameters
        ----------
            attn_map: Tensor
                Attention weight (Key and Query dependent), shape: BHxSxS
            grad: Tensor
                Gradients to apply to the attention map, shape: BHxSxS

        Returns
        -------
            Tensor
                Averaged attention map after applying gradients.
        """
        attn_map = attn_map.reshape(-1, attn_map.shape[-2], attn_map.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        attn_map = grad * attn_map
        attn_map = attn_map.clamp(min=0).mean(dim=0)
        return attn_map

    @staticmethod
    def apply_self_attention_rules(R_ss, cam_ss):
        """
        Rule 6 from Chefer et al.: Applies self-attention rules to update the relevance score.

        Parameters
        ----------
            R_ss: Tensor
                Relevance score matrix.
            cam_ss: Tensor
                Attention map.

        Returns
        -------
            Tensor
                Updated relevance score matrix.
        """
        return torch.matmul(cam_ss, R_ss)

    def generate_relevance(self, transformer_output, category_index=None):
        """
        Generates the relevance score for a given input and category index.

        Parameters
        ----------
            transformer_output
            category_index: List[int], optional
                List of indices to consider in the category mask, default is None.
        """
        #output, _ = self.model(padded_h_node, src_padding_mask, register_hook=True)
        category_mask = torch.ones(transformer_output.size())
        if category_index is not None:
            category_mask = torch.zeros(transformer_output.size())
            category_mask[category_index, :, :] = 1
        loss = (transformer_output * category_mask).sum()
        self.transformer_encoder.zero_grad()
        loss.backward(retain_graph=True)

        num_tokens = self.transformer_encoder.layers[0].get_attn_output().shape[0]

        #I = torch.eye(num_tokens, num_tokens).cuda()
        I = torch.eye(num_tokens, num_tokens)

        for idx, encoder in enumerate(self.transformer_encoder.layers):
            attn_out_weights = encoder.get_attn_output_weights()
            attn_grad = encoder.get_attn_gradients()

            attn_map = self.avg_heads(attn_out_weights, attn_grad)
            #I += self.apply_self_attention_rules(I.cuda(), attn_map.cuda())
            I += self.apply_self_attention_rules(I, attn_map)
        
        return I