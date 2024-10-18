# Transolver (ICML 2024 Spotlight)

:triangular_flag_on_post:**News** (2024.10) Transolver has been integrated into [NVIDIA modulus](https://github.com/NVIDIA/modulus/tree/main/examples/cfd/darcy_transolver).

Transolver: A Fast Transformer Solver for PDEs on General Geometries [[Paper]](https://arxiv.org/abs/2402.02366) [[Slides]](https://wuhaixu2016.github.io/pdf/ICML2024_Transolver.pdf) [[Poster]](https://wuhaixu2016.github.io/pdf/poster_ICML2024_Transolver.pdf)

In real-world applications, PDEs are typically discretized into large-scale meshes with complex geometries. To capture intricate physical correlations hidden under multifarious meshes, we propose the Transolver with the following features:

- Going beyond previous work, Transolver **calculates attention among learned physical states** instead of mesh points, which empowers the model with **endogenetic geometry-general capability**.
- Transolver achieves **22% error reduction over previous SOTA in six standard benchmarks** and excels in **large-scale industrial simulations**, including car and airfoil designs.
- Transolver presents favorable **efficiency, scalability and out-of-distrbution generalizability**.

<p align="center">
<img src=".\pic\Transolver.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of Transolver.
</p>


## Transolver v.s. Previous Transformer Operators

**All of the previous Transformer-based neural operators directly apply attention to mesh points.** However, the massive mesh points in practical applications will cause challenges in both computation cost and capturing physical correlations.

Transolver is based on a more foundational idea, that is **learning intrinsic physical states under complex geometrics**. This design frees our model from superficial and unwieldy meshes and focuses more on physics modeling.

As shown below, **Transolver can precisely capture miscellaneous physical states of PDEs**, such as (a) various fluid-structure interactions in a Darcy flow, (b) different extrusion regions of elastic materials, (c) shock wave and wake flow around the airfoil, (d) front-back surfaces and up-bottom spaces of driving cars.

<p align="center">
<img src=".\pic\physical_states.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> Visualization of learned physical states.
</p>

## Get Started

1. Please refer to different folders for detailed experiment instructions.

2. List of experiments:

- Core code: see [./Physics_Attention.py](https://github.com/thuml/Transolver/blob/main/Physics_Attention.py)
- Standard benchmarks: see [./PDE-Solving-StandardBenchmark](https://github.com/thuml/Transolver/tree/main/PDE-Solving-StandardBenchmark)
- Car design task: see [./Car-Design-ShapeNetCar](https://github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar)
- Airfoil design task: see [./Airfoil-Design-AirfRANS](https://github.com/thuml/Transolver/tree/main/Airfoil-Design-AirfRANS)

## Results

Transolver achieves consistent state-of-the-art in **six standard benchmarks and two practical design tasks**. **More than 20 baselines are compared.**

<p align="center">
<img src=".\PDE-Solving-StandardBenchmark\fig\standard_benchmark.png" height = "300" alt="" align=center />
<br><br>
<b>Table 1.</b> Results on six standard benchmarks.
</p>

<p align="center">
<img src=".\Airfoil-Design-AirfRANS\fig\results.png" height = "300" alt="" align=center />
<br><br>
<b>Table 2.</b> Results on two design tasks: Car and Airfoild design.
</p>

## Showcases

<p align="center">
<img src=".\pic\showcases.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 3.</b> Comparison of Transolver and other models.
</p>

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Contact

If you have any questions or want to use the code, please contact [wuhx23@mails.tsinghua.edu.cn](mailto:wuhx23@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/neuraloperator/neuraloperator

https://github.com/neuraloperator/Geo-FNO

https://github.com/thuml/Latent-Spectral-Models

https://github.com/Extrality/AirfRANS
