import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal



# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class RILI_Autoencoder(nn.Module):
    def __init__(self, state_dim, reward_dim, latent_dim, hidden_dim, timesteps=10):
        super(RILI_Autoencoder, self).__init__()

        self.loss_fcn = nn.MSELoss()
        self.beta = 0.5

        # Strategy encoder (tau^i --> z^i)
        self.enc_zcurr_1 = nn.Linear((state_dim+reward_dim)*timesteps, hidden_dim)
        self.enc_zcurr_2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_zcurr_3 = nn.Linear(hidden_dim, latent_dim)

        # Partner encoder (z1, z2, z3, z4 --> p)
        self.enc_p_1 = nn.Linear(latent_dim*4, hidden_dim)
        self.enc_p_2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_p_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_p_log_var = nn.Linear(hidden_dim, latent_dim)

        # Dynamics (z^i, p^i --> z^i+1)
        self.enc_znext_1 = nn.Linear(latent_dim*2, hidden_dim)
        self.enc_znext_2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_znext_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_znext_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder architecture (states, z --> rewards)
        self.dec1 = nn.Linear(state_dim*timesteps + latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, reward_dim*timesteps)

        self.apply(weights_init_)

    def encoder_strategy(self, tau):
        x = torch.tanh(self.enc_zcurr_1(tau))
        x = torch.tanh(self.enc_zcurr_2(x))
        return self.enc_zcurr_3(x)

    def encoder_partner(self, z1, z2, z3, z4):
        context = torch.cat((z1, z2, z3, z4), -1)
        x = torch.tanh(self.enc_p_1(context))
        x = torch.tanh(self.enc_p_2(x))
        x_mean = self.enc_p_mean(x)
        x_std = torch.exp(0.5 * self.enc_p_log_var(x))
        x = x_mean + x_std * torch.randn_like(x_std)
        return x, x_mean, x_std

    def dynamics(self, z, p):
        context = torch.cat((z, p), -1)
        x = torch.tanh(self.enc_znext_1(context))
        x = torch.tanh(self.enc_znext_2(x))
        z_mean = self.enc_znext_mean(x)
        z_std = torch.exp(0.5 * self.enc_znext_log_var(x))
        z = z_mean + z_std * torch.randn_like(z_std)
        return z, z_mean, z_std

    def encoder(self, tau1, tau2, tau3, tau4):
        z1 = self.encoder_strategy(tau1)
        z2 = self.encoder_strategy(tau2)
        z3 = self.encoder_strategy(tau3)
        z4 = self.encoder_strategy(tau4)
        p, p_mean, p_std = self.encoder_partner(z1, z2, z3, z4)
        z5, z5_mean, z5_std = self.dynamics(z4, p)
        return z5, z5_mean, z5_std, p, p_mean, p_std

    def decoder(self, context):
        x = torch.tanh(self.dec1(context))
        x = torch.tanh(self.dec2(x))
        return (self.dec3(x) - 1.0) * 100

    def forward(self, tau1, tau2, tau3, tau4, states1, rewards1, states2, rewards2, states3, rewards3, states4, rewards4, states5, rewards5):
        z1str = self.encoder_strategy(tau1)
        z2str = self.encoder_strategy(tau2)
        z3str = self.encoder_strategy(tau3)
        z4str = self.encoder_strategy(tau4)
        p, p_mean, p_std = self.encoder_partner(z1str, z2str, z3str, z4str)
        z2hat, z2_mean, z2_std = self.dynamics(z1str, p)
        z3hat, z3_mean, z3_std = self.dynamics(z2str, p)
        z4hat, z4_mean, z4_std = self.dynamics(z3str, p)
        z5hat, z5_mean, z5_std = self.dynamics(z4str, p)
        s1 = torch.flatten(states1, 1)
        s2 = torch.flatten(states2, 1)
        s3 = torch.flatten(states3, 1)
        s4 = torch.flatten(states4, 1)
        s5 = torch.flatten(states5, 1)
        r1str = self.decoder(torch.cat((s1, z1str), 1))
        r2str = self.decoder(torch.cat((s2, z2str), 1))
        r3str = self.decoder(torch.cat((s3, z3str), 1))
        r4str = self.decoder(torch.cat((s4, z4str), 1))
        r2hat = self.decoder(torch.cat((s2, z2hat), 1))
        r3hat = self.decoder(torch.cat((s3, z3hat), 1))
        r4hat = self.decoder(torch.cat((s4, z4hat), 1))
        r5hat = self.decoder(torch.cat((s5, z5hat), 1))
        l1str = self.loss_fcn(r1str, rewards1)
        l2str = self.loss_fcn(r2str, rewards2)
        l3str = self.loss_fcn(r3str, rewards3)
        l4str = self.loss_fcn(r4str, rewards4)
        l2hat = self.loss_fcn(r2hat, rewards2)
        l3hat = self.loss_fcn(r3hat, rewards3)
        l4hat = self.loss_fcn(r4hat, rewards4)
        l5hat = self.loss_fcn(r5hat, rewards5)
        kl_2hat = -0.5 * (1 + torch.log(z2_std.pow(2)) - z2_mean.pow(2) - z2_std.pow(2)).mean()
        kl_3hat = -0.5 * (1 + torch.log(z3_std.pow(2)) - z3_mean.pow(2) - z3_std.pow(2)).mean()
        kl_4hat = -0.5 * (1 + torch.log(z4_std.pow(2)) - z4_mean.pow(2) - z4_std.pow(2)).mean()
        kl_5hat = -0.5 * (1 + torch.log(z5_std.pow(2)) - z5_mean.pow(2) - z5_std.pow(2)).mean()
        kl_p = -0.5 * (1 + torch.log(p_std.pow(2)) - p_mean.pow(2) - p_std.pow(2)).mean()
        curr_loss = l1str+l2str+l3str+l4str
        next_loss = l2hat+l3hat+l4hat+l5hat
        ae_loss = curr_loss + next_loss
        kl_loss = kl_2hat+kl_3hat+kl_4hat+kl_5hat+kl_p
        return ae_loss + self.beta*kl_loss, curr_loss, next_loss, kl_loss
