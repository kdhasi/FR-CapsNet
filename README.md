# FR-CapsNet

Pytorch implementation of the paper "FR-CapsNet: Enhancing Low-Resolution Image Classification via Frequency Routed Capsules." The manuscript is available at XX.

## Dependencies

The codes were executed with the following packages,
- Python 3.9
- Pytorch 2.1.2
- Torchvision 0.16.2
- CUDA 11.8

## Training

 ```
python spatial_baseline_C4K8D24.py --dataset cifar10 --batch_size 64
 ```
 ```
python combined_baseline_C4K8D24.py --dataset cifar10 --batch_size 64
 ```
 ```
python spatial_deep_C4K10D32.py --dataset cifar10 --batch_size 64
 ```
 ```
python combined_deep_C4K10D32.py --dataset cifar10 --batch_size 64
 ```

## Datasets

- MNIST
- SVHN
- CIFAR-10
- CIFAR-100

## Performance

<table>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">CIFAR10</th>
    <th colspan="2">SVHN</th>
  </tr>
  <tr>
    <th>(8 x 8)</th>
    <th>(32 x 32)</th>
    <th>(8 x 8)</th>
    <th>(32 x 32)</th>
  </tr>
  <tr>
    <td>S-baseline</td>
    <td align="center">75.41</td>
    <td align="center">90.55</td>
    <td align="center"><b>94.75</b></td>
    <td align="center">96.06</td>
  </tr>
  <tr>
    <td>C-baseline</td>
    <td align="center">75.69</td>
    <td align="center">90.96</td>
    <td align="center">93.79</td>
    <td align="center">96.24</td>
  </tr>
  <tr>
    <td>S-deep</td>
    <td align="center">78.54</td>
    <td align="center">92.51</td>
    <td align="center">94.41</td>
    <td align="center"><b>97.08</b></td>
  </tr>
  <tr>
    <td>C-deep</td>
    <td align="center"><b>78.90</b></td>
    <td align="center">91.84</td>
    <td align="center"><b>94.82</b></td>
    <td align="center">97.02</td>
  </tr>
  <tr>
    <td>S-deep (transfer learned)</td>
    <td align="center"><b>79.91</b></td>
    <td align="center">93.03</td>
    <td align="center">--</td>
    <td align="center">--</td>
  </tr>
  <tr>
    <td>C-deep (transfer learned)</td>
    <td align="center">79.53</td>
    <td align="center">93.10</td>
    <td align="center">--</td>
    <td align="center">--</td>
  </tr>
</table>

## Citation

If you find this work useful, please cite the following paper:
