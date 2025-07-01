import casperfpga
import numpy as np
from matplotlib import pyplot as plt
import struct
import time
from argparse import ArgumentParser

FIXED_LEN = 30

def Scope(darray):
    try:
        plt.close('all')
    except:
        pass
    fig = plt.figure()
    subfig = fig.add_subplot(111)
    for d in darray:
        subfig.plot(d)
    subfig.set_title('Time Domain Data')
    subfig.set_xlabel('Samples')
    subfig.grid(True)

def CreateTestFilter(dac_fft_len, fft_points, bw=1):
    N = dac_fft_len//2
    fil = np.zeros(N)
    L = int(N*bw)
    passband = np.zeros(L)
    factor = dac_fft_len//fft_points/2
    for i in range(L):
        # we need to filter DC
        if i == 0:
            continue
        elif i%factor == 0:
            passband[i] = 1
    fil[:L] = passband
    return fil
    
    
def DacDataGen(channel_num, cfil, single_bin=0, scale=2**13):
    N = channel_num
    M = (N//2)-1
    fil = np.zeros(M+1)
    if single_bin == 0:
        fil = cfil
    else:
        fil[single_bin] = 1
    d_freq = np.zeros(N) + 1j*np.zeros(N)
    d_freq_half0 = np.exp(1j*np.random.uniform(0, 2*np.pi, size=M)) * fil[1:M+1]
    d_freq_half1 = np.conjugate(d_freq_half0)[::-1]
    d_freq[1:M+1] = d_freq_half0
    d_freq[-M:] = d_freq_half1
    d_time = np.fft.ifft(d_freq)
    d_time_int = np.round((d_time*scale))
    return d_time_int

def ADCSampling(data, adc_fs=500, dac_fs=1000):
    decimation_factor = int(dac_fs/adc_fs)
    tmp = data.reshape(-1, decimation_factor)
    return tmp[:,0]

def LPFilter(data, bw=0.5):
    l = int(len(data)/2)
    fil = np.zeros(l*2)
    passband_len = int(2*l*bw)
    passband = np.ones(passband_len)
    # we don't need DC
    passband[0] = 0
    fil[:passband_len] = passband
    tmp = fil[:l][::-1]
    fil[-l:] = tmp
    data = data * fil
    return data
    
def SpectraAnalyzer(data, fs=1000, show_bin=False):
    N = len(data)
    spec = np.fft.fft(data)
    spec_shift = np.fft.fftshift(spec)
    freq = np.fft.fftfreq(N, 1/fs)
    freq_shift = np.fft.fftshift(freq)
    try:
        plt.close('all')
    except:
        pass
    fig = plt.figure()
    subfig = fig.add_subplot(111)
    if show_bin == True:
        s = np.abs(spec_shift[-int(N/2):])
        subfig.plot(np.arange(len(s)),s)
        subfig.set_xlabel('Bins')
    else:
        subfig.plot(freq_shift[-int(N/2):], abs(spec_shift[-int(N/2):]))
        subfig.set_xlabel('MHz')
    subfig.set_title('Spectra Data')
    subfig.set_yscale('log')
    subfig.grid(True)
    return spec

def ToBytes(data):
    N = len(data)
    data = np.short(data)
    buf = struct.pack('>%dh'%N, *data)
    return buf

def print_parameters(args):
    print('**************************************')
    print('%s: %s'%('ADC Sampling Rate(MSps):'.ljust(FIXED_LEN), args.adcfs))
    print('%s: %s'%('ADC FFT points:'.ljust(FIXED_LEN), args.adcfft))
    print('%s: %s'%('DAC Sampling Rate(MSps):'.ljust(FIXED_LEN), args.dacfs))
    print('%s: %s'%('DAC iFFT points:'.ljust(FIXED_LEN), args.daclen))
    print('%s: %s'%('Bandwidth(MHz)'.ljust(FIXED_LEN), args.bw))
    print('%s: %s'%('Single Bin'.ljust(FIXED_LEN), args.sbin))

def main():
    parser = ArgumentParser(description="Usage for Setting DAC on RFSoC2x2.")
    parser.add_argument('--ip',dest='ip', type=str, default='127.0.0.1',help='The IP address of the RFSoC2x2')
    parser.add_argument('--fpg',dest='fpg', type=str, default='rfsocdactut_2025-06-30_1113.fpg', help='The fpg file uploaded to the board.')
    parser.add_argument('--single-bin', dest='sbin', type=int, default=0, help='The bin number of the signal. 0 means all bins.')
    parser.add_argument('--bw', dest='bw', type=float, default=250.0, help='The bandwidth of the signal in MHz.')
    parser.add_argument('--dac-len', dest='daclen', type=int, default=32768, help='The DAC data length.')
    parser.add_argument('--adc-fft', dest='adcfft', type=int, default=2048, help='FFT points on the ADC samples.')
    parser.add_argument('--adc-fs', dest='adcfs', type=float, default=500.0, help='ADC sampling rate in MSps.')
    parser.add_argument('--dac-fs', dest='dacfs', type=float, default=1000.0, help='DAC sampling rate in MSps.')
    parser.add_argument('--scale', dest='scale', type=str, default='2**20', help='The DAC output scale.')
    parser.add_argument('--npz', dest='npz', type=str, default=None, help='The npz file, which stores the data written into DAC. The key has to be `data`.')
    parser.add_argument('--skip-config', dest='skipconfig', action='store_true', default=False, help='Skip the FPGA and PLL config.')
    args = parser.parse_args()
    # config the rfosc2x2
    print('*******************************************')
    print('Conneting to RFSoC2x2 at %s...'%args.ip)
    rfsoc=casperfpga.CasperFpga(args.ip, transport=casperfpga.KatcpTransport)
    if not args.skipconfig:
        print('*******************************************')
        print('Configuring FPGA with %s...'%args.fpg)
        rfsoc.upload_to_ram_and_program(args.fpg)
        print('*******************************************')
        print('Configuring PLLs...')
        rfdc = rfsoc.adcs['rfdc']
        ## init rfdc
        rfdc.init()
        ## init LMK and LMX
        c = rfdc.show_clk_files()
        # ref = 12.5MHz
        ## rfdc.progpll('lmk', c[0])
        # ref = 250MHz
        rfdc.progpll('lmk', c[1])
        rfdc.progpll('lmx', c[3])
        time.sleep(1)
        rfdc.status()
    # Generate data for DAC
    ## Check if we already have data file
    if args.npz is not None:
        print('*******************************************')
        print('Load data from npz file')
        print('%s: %s'%('File Name'.ljust(10), args.npz))
        dfiles = np.load(args.npz)
        data = dfiles['data']
        max_addr = dfiles['max_addr']
        print('%s: %d'%('Samples'.ljust(10), len(data)))
        print('%s: %d'%('Max Addr'.ljust(10), max_addr))
    else:
        # adc sampling rate
        adc_fs = args.adcfs
        # dac sampling rate
        dac_fs = args.dacfs
        # bandwidth
        bw = args.bw
        # fft points on the SNAP
        fft_points = args.adcfft
        # check the single bin
        # if it's 0, the signal will show up in all bins
        single_bin=args.sbin
        # fixed parameters in FPGA design
        bytes_per_axis = 8
        bytes_per_sample = 2
        # dac FFT length
        dac_fft_len = args.daclen
        scale = eval(args.scale)
        print_parameters(args)
        single_bin = int(single_bin*dac_fft_len//fft_points//(dac_fs//adc_fs))
        # generate the data for the DAC with DAC sampling rate=1000
        cfil = CreateTestFilter(dac_fft_len, fft_points, bw = bw/dac_fs*2)
        d_time = DacDataGen(dac_fft_len, cfil, single_bin=single_bin, scale = scale)
        data = d_time.real
        samples_per_cyc = dac_fft_len
        max_addr = int(samples_per_cyc*bytes_per_sample/bytes_per_axis - 3)
        rfsoc.write_int('wf_en', 1)
        print('%s: %s'%('Max Addr'.ljust(FIXED_LEN), max_addr))
        print('%s: %s'%('Peak Value'.ljust(FIXED_LEN), max(d_time.real)))
    # write data into sbram for the DAC
    nbuf = ToBytes(data)
    zeros = np.zeros(2**13*2, dtype=np.short)
    zeros_bytes = struct.pack('>%dh'%len(zeros), *zeros)
    rfsoc.write('wf_bram_0', zeros_bytes)
    rfsoc.write('wf_bram_0', nbuf)
    rfsoc.write_int('addr_max', max_addr)
    rfsoc.write_int('wf_en', 1)
    print('*******************************************')
    print('* Note: There are 2 clks delay in the max_addr')
    print('        comparison in the FPGA design, so the')
    print('        max_addr should be subtracted by 2.')
    print('*******************************************')
    print('Done!')

if __name__ == '__main__':
    main()