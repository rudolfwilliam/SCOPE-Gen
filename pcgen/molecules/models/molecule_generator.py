"""Class taken from https://github.com/cvignac/DiGress . We adapted it to masking with a scaffold, because we could not find this implemented in the original code."""

from pcgen.models.base import GenerativeModel
from pcgen.molecular_extensions.data.base import squeeze
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion import diffusion_utils
from src import utils
from torch.nn import functional as F
from torch import nn
import torch


class DiGressMoleculeGenerator(DiscreteDenoisingDiffusion, GenerativeModel):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, guidance_model=None):
        self.guidance_model = guidance_model
        super().__init__(cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                         domain_features)

    @torch.no_grad()
    def sample(self, num_samples, cond=None):
        """
        :param num_samples: int
        :cond: a tuple (scaffold_mask, num_nodes), where scaffold_mask is a tuple of tensors (X, E) of size (n), (n, n) (optional)
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        scaffold_mask, num_nodes = cond
        scaffold_mask = squeeze(scaffold_mask)
        # convert scaffold mask to one-hot
        if scaffold_mask is not None:
            X_scaff_mask = F.one_hot(scaffold_mask[0], num_classes=self.Xdim_output).float()
            E_scaff_mask = F.one_hot(scaffold_mask[1], num_classes=self.Edim_output).float()
            X_scaff_mask_filled = torch.zeros((num_samples, num_nodes, self.Xdim_output), device=self.device)
            E_scaff_mask_filled = torch.zeros((num_samples, num_nodes, num_nodes, self.Edim_output), device=self.device)
            X_scaff_mask_filled[:, :X_scaff_mask.shape[0], :] = X_scaff_mask
            E_scaff_mask_filled[:, :E_scaff_mask.shape[0], :E_scaff_mask.shape[1], :] = E_scaff_mask
            node_mask_scaff = torch.zeros((num_samples, num_nodes), device=self.device, dtype=torch.bool)
            node_mask_scaff[:, :X_scaff_mask.shape[0]] = True
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(num_samples, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(num_samples, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(num_samples, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        #if scaffold_mask is not None:
        #X[:, :scaffold_mask[0].shape[0], :] = X_scaff_mask
        #E[:, :scaffold_mask[1].shape[0], :scaffold_mask[1].shape[1], :] = E_scaff_mask
        # sample the scaffold mask
            
        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((num_samples, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s, predicted_graph = self.sample_p_zs_given_zt(t_norm, X, E, y, node_mask,
                                                                                       last_step=s_int == 100)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
            if scaffold_mask is not None:
                scaff_mask_noisy = self.apply_noise_given_t(X_scaff_mask_filled, E_scaff_mask_filled, y, node_mask_scaff, s_int)
                X[:, :scaffold_mask[0].shape[0], :] = scaff_mask_noisy["X_t"][:, :scaffold_mask[0].shape[0], :]
                E[:, :scaffold_mask[1].shape[0], :scaffold_mask[1].shape[1], :] = scaff_mask_noisy["E_t"][:, :scaffold_mask[1].shape[0], :scaffold_mask[1].shape[1], :]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        molecule_list = []
        for i in range(num_samples):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append((atom_types, edge_types))

        return molecule_list
    
    @torch.no_grad()
    def apply_noise_given_t(self, X, E, y, node_mask, t):
        """ Sample noise and apply it to the data. """
        t = torch.tensor([t])
        assert t >= 0 and t <= self.T
        s_int = t - 1

        t_float = t / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    """def cond_fn(self, noisy_data, node_mask, target=None):
        #self.guidance_model.eval()
        loss = nn.MSELoss()

        t = noisy_data['t']

        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['t']

        with torch.enable_grad():
            x_in = X.float().detach().requires_grad_(True)
            e_in = E.float().detach().requires_grad_(True)

            pred = self.guidance_model.model(x_in, e_in, y, node_mask)

            # normalize target
            target = target.type_as(x_in)

            mse = loss(pred.y, target.repeat(pred.y.size(0), 1))

            t_int = int(t[0].item() * 500)
            if t_int % 10 == 0:
                print(f'Regressor MSE at step {t_int}: {mse.item()}')

            # calculate gradient of mse with respect to x and e
            grad_x = torch.autograd.grad(mse, x_in, retain_graph=True)[0]
            grad_e = torch.autograd.grad(mse, e_in)[0]

            x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
            bs, n = x_mask.shape[0], x_mask.shape[1]

            e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
            diag_mask = torch.eye(n)
            diag_mask = ~diag_mask.type_as(e_mask1).bool()
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

            mask_grad_x = grad_x * x_mask
            mask_grad_e = grad_e * e_mask1 * e_mask2 * diag_mask

            mask_grad_e = 1 / 2 * (mask_grad_e + torch.transpose(mask_grad_e, 1, 2))
            return mask_grad_x, mask_grad_e

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, input_properties):
        Samples from zs ~ p(zs | zt). Only used during sampling.
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        # # Guidance
        lamb = self.args.guidance.lambda_guidance

        grad_x, grad_e = self.cond_fn(noisy_data, node_mask, input_properties)

        p_eta_x = torch.softmax(- lamb * grad_x, dim=-1)
        p_eta_e = torch.softmax(- lamb * grad_e, dim=-1)

        prob_X_unnormalized = p_eta_x * prob_X
        prob_X_unnormalized[torch.sum(prob_X_unnormalized, dim=-1) == 0] = 1e-7
        prob_X = prob_X_unnormalized / torch.sum(prob_X_unnormalized, dim=-1, keepdim=True)

        prob_E_unnormalized = p_eta_e * prob_E
        prob_E_unnormalized[torch.sum(prob_E_unnormalized, dim=-1) == 0] = 1e-7
        prob_E = prob_E_unnormalized / torch.sum(prob_E_unnormalized, dim=-1, keepdim=True)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)"""
