%%
load('epochs.mat')
subplot(1, 3, 1)
hold on
plot(discriminator_loss)
ylabel("loss")
xlabel("iterations (per batch)")
title("Discriminator loss")
hold off
subplot(1, 3, 2)
hold on
plot(generator_loss)
ylabel("loss")
xlabel("iterations (per batch)")
title("Generator loss")
hold off
subplot(1, 3, 3)
hold on
plot(reconstruction_loss)
ylabel("loss")
xlabel("iterations (per batch)")
title("Reconstruction loss")
hold off

%%