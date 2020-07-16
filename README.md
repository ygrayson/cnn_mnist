# 卷积神经网络(CNN)识别MNIST数据集

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/ygrayson/cnn_mnist">
    <img src="images/sample_digit.png" alt="Digit" width="80" height="80">
  </a>

  <h3 align="center">卷积神经网络CNN识别手写数字MNIST数据集</h3>
  <p align="center">
    机器学习界的果蝇实验，欢迎参考
  </p>
</p>


<!-- TABLE OF CONTENTS -->
## 目录

* [简介](#简介)
  * [使用工具](#使用工具)
* [安装](#安装)
  * [语言](#语言)
  * [平台](#平台)
* [开始识别数字！](#开始识别数字吧！)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## 简介

深度学习发展迅速，MNIST手写数字数据集作为机器学习早期的数据集已经被公认为是机器学习界的果蝇实验（Hinton某年)，卷积神经网络是识别图像非常有效的一种架构，于是用<b>CNN识别手写数字</b>也就成为了机器学习界的经典实验。在这个repo中我会呈现最基本的CNN识别MNIST数据集过程。
主要步骤如下：
* 载入数据
* 看看数据啥样儿（也即Visualization）
* 建立模型
* 训练模型
* 测试模型（看看训练的好不好）


### 使用工具
主要使用的语言和平台如下：
* 语言：[Python](https://www.python.org/)
* 平台：[Pytorch](https://pytorch.org/)


<!-- GETTING STARTED -->
## 安装

安装包括Python语言和几个必备的Pytorch包，使用Linux系统的安装方式如下:

### 语言
检查一下Python语言：
* Python
```sh
$ python3 --version
Python 3.7.7
```

### Pytorch包
1. 安装Pytorch
```sh
pip install torch
```
2. 安装Torchvision
```sh
pip install torchvision
```

3. 安装Matplotlib
```sh
pip install matplotlib
```


<!-- USAGE EXAMPLES -->
## 开始识别数字吧！

完成了安装，我们就可以开始训练神经网络识别数字了，首先载入数据。


<!-- ROADMAP -->
## 载入数据

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Qianbo Yin - [@LinkedIn](https://www.linkedin.com/in/qianbo-yin-a91861114/)

Project Link: [https://github.com/ygrayson/cnn_mnist](https://github.com/ygrayson/cnn_mnist)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* README.md created with [Best-README-Template](https://github.com/othneildrew/Best-README-Template)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
