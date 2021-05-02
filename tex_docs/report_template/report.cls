% Latex2e Class
% ----------------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{report}[30/04/2021 MS. Project Report ]


% Report Layout
% ----------------------------------------------------------------------------
\LoadClass[12pt, oneside]{book} % 12 pt font, two-sided book style
\RequirePackage[a4paper,total={155mm, 240mm}, top=30mm, ]{geometry} 
	
\RequirePackage[bf]{caption} % caption: FIG in bold
\RequirePackage{emptypage} % Blank Pages


% Spacing
% ............................................................................
\newcommand{\thesisspacing}{\singlespacing}

% Numbered/ Un-numbered Chapters
% ............................................................................
\newcommand{\numberedchapter}{ % Preparation for numbered chapters
	\cleardoublepage % To make sure the previous headers are passed
	\fancyhead[RE]{{\bfseries \leftmark}}% Headers for left pages
	\fancyhead[LO]{{\bfseries \rightmark}}
	}% Headers for right pages
	
\newcommand{\unnumberedchapter}[1]{ % Preparation for unnumbered chapters
	\cleardoublepage % To make sure the previous headers are passed
	\phantomsection % To help hyperref link to the right page
	\addcontentsline{toc}{chapter}{#1} % Also adds the chapter name to the Contents
	\fancyhead[RE]{{\Large \bfseries }} % Headers for left pages
	\fancyhead[LO]{}
	}%Headers for right pages


\newcommand{\startpreamble}{
	\pagestyle{fancy} % Changes the headers
	\fancyhf{}% Clears header and footer
	\renewcommand{\chaptermark}[1]{\markboth{##1}{}} % Getting the chapter name right
	\renewcommand{\sectionmark}[1]{\markright{\thesection\; ##1}} % Getting the section name right
	\fancyhead[RO,LE]{\thepage} % page number on the outside of headers
	\setcounter{page}{2} % Include the title page in page counting
	\thesisspacing % Double spacing
	}

