filenames = ["220.wav", "440.wav", "440cut.wav", "774.wav", "a_C3_ep44.wav", "a_C4_ugp44.wav", "karol_a_1.wav", "karol_a_2.wav"]
frequencies = []

for name = filenames
    name
    [s,Fs] = audioread(name);


    len = min(1000, length(s));
    scut = s(1:len);
    xc = xcorr(scut);
    xccut=xc(length(scut):end);

    [pks, locs]=findpeaks(xccut);
    [peak_value, pi] = max(pks);
    peak_index = locs(pi);
    f0 = Fs/peak_index
    frequencies = [frequencies, f0]

    fig = figure();

    title("220")
    subplot(2,1,1)
    plot(scut)
    title('Sygna³')
    subplot(2,1,2)
    hold on
    plot(xccut)
    scatter(locs, pks)
    scatter(peak_index, peak_value, 'filled', 'bo')
    hold off
    title('Autokorelacja')

    outputfile = append(name, '.eps');
    outputfile = append('output_', outputfile);
    saveas(fig, outputfile, 'eps')
end
