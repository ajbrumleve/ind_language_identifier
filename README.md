# Indonesian Language Identifier
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



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
<div align="center">

  <h3 align="center">Indonesian Language Identifier</h3>

  <p align="center">
    Identify the Indonesian language being used
    <br />
    <a href="https://github.com/ajbrumleve/ind_language_identifier/issues">Report Bug</a>
    ·
    <a href="https://github.com/ajbrumleve/ind_language_identifier/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This code is a piece of the quality control of the ikata app developed by Intek Solutions. The ikata app crowdsources language data collection in Indonesia. A need in the app was to be able to confirm the language being input into the app was the expected language. This code currently trains a Naive Bayes model on labelled Indonesian, English, and Alas sentences. Alas is a language spoken in Aceh Tenggara in Northern Sumatra. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With



* [![Python][Python]][Python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

Require packages wxPython, numpy, pandas, scikit-learn, seaborn, and matpotlib.

Use `pip` to install the packages from PyPI:

```bash
pip install wxPython
pip install numpy
pip install pandas
pip install scikit-learn
pip install seaborn
pip install matplotlib
```

### Installation

1. Download and unzip [this entire repository from GitHub](https://github.com/ajbrumleve/ind_language_identifier), either interactively, or by entering the following in your Terminal.
    ```bash
    git clone https://github.com/ajbrumleve/ind_language_identifier.git
    ```
2. Navigate into the top directory of the repo on your machine
    ```bash
    cd house_price_prediction
    ```
3. Create a virtualenv and install the package dependencies. If you don't have `pipenv`, you can follow instructions [here](https://pipenv.pypa.io/en/latest/install/) for how to install.
    ```bash
    pipenv install
    ```
4. Run `main.py` to run the graphical interface. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

When you run main.py, a graphical interface will appear which lets you input a sentence. If a model has not been trianed before it will quickly train the model and save it. Future runs will not require this step. The output will give the most likely language, how closely the trigrams in the input text match the model for that language, and the probabilities for each language. If the input text is not sufficiently similar to the languages used in training, the model will output unknown. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Some ideas of ways to extend this code include:
 - Add date from other Indonesian languages
 - Add other classification methods - eg Markov model

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Andrew Brumleve - [@AndrewBrumleve](https://twitter.com/AndrewBrumleve) - ajbrumleve@gmail.com

Project Link: [https://github.com/ajbrumleve/ind_language_identifier](https://github.com/ajbrumleve/ind_language_identifier)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ajbrumleve/ind_language_identifier.svg?style=for-the-badge
[contributors-url]: https://github.com/ajbrumleve/ind_language_identifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ajbrumleve/ind_language_identifier.svg?style=for-the-badge
[forks-url]: https://github.com/ajbrumleve/ind_language_identifier/network/members
[stars-shield]: https://img.shields.io/github/stars/ajbrumleve/ind_language_identifier.svg?style=for-the-badge
[stars-url]: https://github.com/ajbrumleve/ind_language_identifier/stargazers
[issues-shield]: https://img.shields.io/github/issues/ajbrumleve/ind_language_identifier.svg?style=for-the-badge
[issues-url]: https://github.com/ajbrumleve/ind_language_identifier/issues
[license-shield]: https://img.shields.io/github/license/ajbrumleve/ind_language_identifier.svg?style=for-the-badge
[license-url]: https://github.com/ajbrumleve/ind_language_identifier/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: (https://www.linkedin.com/in/andrew-brumleve-574239227/)
[product-screenshot]: images/screenshot.png
[Python]:  	https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org/

