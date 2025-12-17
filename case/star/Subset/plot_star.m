clc
clear
close all
load Subset_DIC_001.mat


figure(1)

[M, N] = size(DIC_result.plot_v);
set(gcf, 'Units', 'pixels', 'Position', [100 100 N M]);

alpha_channel = DIC_result.plot_validpoints * 255;

h1 = imagesc(DIC_result.plot_v);
colormap jet;
caxis([-1, 1]);

set(h1, 'AlphaData', alpha_channel, ...
        'AlphaDataMapping', 'direct', ...
        'Interpolation', 'nearest');

axis image;
axis off;
set(gca, 'Position', [0 0 1 1]);
