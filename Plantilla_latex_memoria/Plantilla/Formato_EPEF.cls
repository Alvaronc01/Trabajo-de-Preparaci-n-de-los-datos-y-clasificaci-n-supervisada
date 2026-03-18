% Identificación
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Formato_EPEF}[2020/11/10 Plantilla para Trabajos Finales]

% Definimos la clase principal del documento
\DeclareOption{oneside}{\PassOptionsToClass{oneside}{book}}
\DeclareOption{twoside}{\PassOptionsToClass{twoside}{book}}
\ProcessOptions\relax
\LoadClass[12pt,openright,twoside,a4paper]{book}

% Definición de los paquetes utilizados
\RequirePackage[utf8]{inputenc}
\RequirePackage[a4paper,left=35mm,top=25mm,bottom=25mm,right=20mm]{geometry}
\RequirePackage[USenglish,spanish,es-nodecimaldot,es-tabla]{babel}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{titlesec}
\RequirePackage[shortlabels]{enumitem}
\RequirePackage{tocloft}
\RequirePackage{etoc}
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{amsmath}
\RequirePackage{amstext}
\RequirePackage{etoolbox}
\RequirePackage{subfiles}
\RequirePackage{fancyhdr}
\RequirePackage{subcaption}
\RequirePackage{multirow}
\RequirePackage{hyperref}
\RequirePackage{longtable}
\RequirePackage{pdfpages}
\RequirePackage[resetlabels,labeled]{multibib}
\RequirePackage{listings}

% Definición del color "Rosa UDC"
\definecolor{UDC_Pink}{RGB}{199, 0, 125}

% Cambio del tipo de fuente para el documento
\renewcommand{\familydefault}{\sfdefault} % Sans Serif
\usepackage{lmodern}

% Modificación para el título de los capítulos
\newcommand{\unchapter}[1]{
    \begingroup
    \let\@makechapterhead\@gobble
    \chapter{#1}
    \endgroup
}

% Modificación del formato hoja
%\renewcommand{\baselinestretch}{1.25}% interlineado
\headsep 7.5mm
\topmargin -5mm
\textheight 23.45cm
\textwidth 16cm
\oddsidemargin 0mm
\evensidemargin 0mm
\footskip 10mm

% Modificación de las cabeceras y pies de página
\fancypagestyle{plain}{
    \fancyhf{}
    \renewcommand{\headrulewidth}{1pt}
    \renewcommand{\footrulewidth}{1pt}
    \fancyhead[LE,LO]{\EscuelaAbrevidado}
    \fancyhead[CE,CO]{\Titulacion}
    \fancyhead[RE,RO]{\TipoAbreviado}
    \fancyfoot[LE,LO]{\Fecha}
    \fancyfoot[CE,CO]{\NombreDocumento}
    \fancyfoot[RE,RO]{\thepage}
}
\pagestyle{plain}

% Modificaciones para las tablas de contenido
\setcounter{secnumdepth}{4}
\setcounter{tocdepth}{4}
\renewcommand{\contentsname}{\hfill Índice \hfill}
\cftsetindents{chapter}{0em}{2em}
\cftsetindents{section}{0em}{2.5em}
\cftsetindents{subsection}{0em}{3em}
\cftsetindents{subsubsection}{0em}{3.5em}
\cftsetindents{paragraph}{0em}{4em}

% Modificación del entorno de la bibliografía
\newcites{Norma}{Normas y disposiciones legales}
\newcites{RefFig}{Referencias de las figuras}

\makeatletter
\renewenvironment{thebibliography}[1]{
    \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}
    \list{\@biblabel{\@arabic\c@enumiv}}{
        \settowidth\labelwidth{\@biblabel{#1}}
        \leftmargin\labelwidth
        \advance\leftmargin\labelsep
        \@openbib@code
        \usecounter{enumiv}
        \let\p@enumiv\@empty
        \renewcommand\theenumiv{\@arabic\c@enumiv}
        }
    \sloppy
    \clubpenalty4000
    \@clubpenalty \clubpenalty
    \widowpenalty4000
    \sfcode`\.\@m
    }{
    \def\@noitemerr{
        \@latex@warning{Empty `thebibliography' environment}
        }
    \endlist
    }
\makeatother

% Definición de los comandos usados a lo largo del documento
\newcommand{\NombreDocumento}{Índice general}
\renewcommand{\chaptername}{Índice general}

% Definición del "lenguaje de programación" Arduino para cargar código generado en este programa
\definecolor{arduinoGreen}{rgb}{0.17,0.43,0.01}
\definecolor{arduinoGrey}{rgb}{0.47,0.47,0.33}
\definecolor{arduinoOrange}{rgb}{0.8,0.4,0.0}
\definecolor{arduinoBlue}{rgb}{0.01,0.61,0.98}
\definecolor{arduinoDarkBlue}{rgb}{0.0,0.2,0.5}

\lstdefinelanguage{Arduino}{
    language=C++,
    keywordstyle=\color{arduinoGreen},
    deletekeywords={
        break, case, override, final, continue, default, do, else, for, if, return, goto, switch, throw,
        try, while, setup, loop, export, not, or, and, xor, include, define, elif, else, error, if, ifdef,
        ifndef, pragma, warning, HIGH, LOW, INPUT, INPUT_PULLUP, OUTPUT, DEC, BIN, HEX, OCT, PI, HALF_PI,
        TWO_PI, LSBFIRST, MSBFIRST, CHANGE, FALLING, RISING, DEFAULT, EXTERNAL, INTERNAL, INTERNAL1V1,
        INTERNAL2V56, LED_BUILTIN, LED_BUILTIN_RX, LED_BUILTIN_TX, DIGITAL_MESSAGE, FIRMATA_STRING,
        ANALOG_MESSAGE, REPORT_DIGITAL, REPORT_ANALOG, SET_PIN_MODE, SYSTEM_RESET, SYSEX_START, auto,
        int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, char16_t, char32_t,
        operator, enum, delete, bool, boolean, byte, char, const, false, float, double, null, NULL, int,
        long, new, private, protected, public, short, signed, static, volatile, String, void, true,
        unsigned, word, array, sizeof, dynamic_cast, typedef, const_cast, struct, static_cast, union,
        friend, extern, class, reinterpret_cast, register, explicit, inline, _Bool, complex, _Complex,
        _Imaginary, atomic_bool, atomic_char, atomic_schar, atomic_uchar, atomic_short, atomic_ushort,
        atomic_int, atomic_uint, atomic_long, atomic_ulong, atomic_llong, atomic_ullong, virtual, PROGMEM,
        Serial, Serial1, Serial2, Serial3, SerialUSB, Keyboard, Mouse, abs, acos, asin, atan, atan2, ceil,
        constrain, cos, degrees, exp, floor, log, map, max, min, radians, random, randomSeed, round, sin,
        sq, sqrt, tan, pow, bitRead, bitWrite, bitSet, bitClear, bit, highByte, lowByte, analogReference,
        analogRead, analogReadResolution, analogWrite, analogWriteResolution, attachInterrupt,
        detachInterrupt, digitalPinToInterrupt, delay, delayMicroseconds, digitalWrite, digitalRead,
        interrupts, millis, micros, noInterrupts, noTone, pinMode, pulseIn, pulseInLong, shiftIn, shiftOut,
        tone, yield, Stream, begin, end, peek, read, print, println, available, availableForWrite, flush,
        setTimeout, find, findUntil, parseInt, parseFloat, readBytes, readBytesUntil, readString,
        readStringUntil, trim, toUpperCase, toLowerCase, charAt, compareTo, concat, endsWith, startsWith,
        equals, equalsIgnoreCase, getBytes, indexOf, lastIndexOf, length, replace, setCharAt, substring,
        toCharArray, toInt, press, release, releaseAll, accept, click, move, isPressed, isAlphaNumeric,
        isAlpha, isAscii, isWhitespace, isControl, isDigit, isGraph, isLowerCase, isPrintable, isPunct,
        isSpace, isUpperCase, isHexadecimalDigit,
    },
    morekeywords={
        break, case, override, final, continue, default, do, else, for, if, return, goto, switch, throw,
        try, while, setup, loop, export, not, or, and, xor, include, define, elif, else, error, if, ifdef,
        ifndef, pragma, warning,
    },
    keywordstyle=[2]\color{arduinoBlue},
    keywords=[2]{
        HIGH, LOW, INPUT, INPUT_PULLUP, OUTPUT, DEC, BIN, HEX, OCT, PI, HALF_PI, TWO_PI, LSBFIRST,
        MSBFIRST, CHANGE, FALLING, RISING, DEFAULT, EXTERNAL, INTERNAL, INTERNAL1V1, INTERNAL2V56,
        LED_BUILTIN, LED_BUILTIN_RX, LED_BUILTIN_TX, DIGITAL_MESSAGE, FIRMATA_STRING, ANALOG_MESSAGE,
        REPORT_DIGITAL, REPORT_ANALOG, SET_PIN_MODE, SYSTEM_RESET, SYSEX_START, auto, int8_t, int16_t,
        int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, char16_t, char32_t, operator, enum, delete,
        bool, boolean, byte, char, const, false, float, double, null, NULL, int, long, new, private,
        protected, public, short, signed, static, volatile, String, void, true, unsigned, word, array,
        sizeof, dynamic_cast, typedef, const_cast, struct, static_cast, union, friend, extern, class,
        reinterpret_cast, register, explicit, inline, _Bool, complex, _Complex, _Imaginary, atomic_bool,
        atomic_char, atomic_schar, atomic_uchar, atomic_short, atomic_ushort, atomic_int, atomic_uint,
        atomic_long, atomic_ulong, atomic_llong, atomic_ullong, virtual, PROGMEM,
    },
    keywordstyle=[3]\bfseries\color{arduinoOrange},
    keywords=[3]{
        Serial, Serial1, Serial2, Serial3, SerialUSB, Keyboard, Mouse,
    },
    keywordstyle=[4]\color{arduinoOrange},
    keywords=[4]{
        abs, acos, asin, atan, atan2, ceil, constrain, cos, degrees, exp, floor, log, map, max, min,
        radians, random, randomSeed, round, sin, sq, sqrt, tan, pow, bitRead, bitWrite, bitSet, bitClear,
        bit, highByte, lowByte, analogReference, analogRead, analogReadResolution, analogWrite,
        analogWriteResolution, attachInterrupt, detachInterrupt, digitalPinToInterrupt, delay,
        delayMicroseconds, digitalWrite, digitalRead, interrupts, millis, micros, noInterrupts, noTone,
        pinMode, pulseIn, pulseInLong, shiftIn, shiftOut, tone, yield, Stream, begin, end, peek, read,
        print, println, available, availableForWrite, flush, setTimeout, find, findUntil, parseInt,
        parseFloat, readBytes, readBytesUntil, readString, readStringUntil, trim, toUpperCase, toLowerCase,
        charAt, compareTo, concat, endsWith, startsWith, equals, equalsIgnoreCase, getBytes, indexOf,
        lastIndexOf, length, replace, setCharAt, substring, toCharArray, toInt, press, release, releaseAll,
        accept, click, move, isPressed, isAlphaNumeric, isAlpha, isAscii, isWhitespace, isControl, isDigit,
        isGraph, isLowerCase, isPrintable, isPunct, isSpace, isUpperCase, isHexadecimalDigit,
    },
    stringstyle=\color{arduinoDarkBlue},
    commentstyle=\color{arduinoGrey},
    numbers=left,
    numbersep=5pt,
    numberstyle=\color{arduinoGrey},
    breaklines=true,
    tabsize=2,
    basicstyle=\ttfamily,
    literate={á}{{\'a}}1  {é}{{\'e}}1  {í}{{\'i}}1 {ó}{{\'o}}1  {ú}{{\'u}}1 {Á}{{\'A}}1  {É}{{\'E}}1  {Í}{{\'I}}1 {Ó}{{\'O}}1  {Ú}{{\'U}}1 {ñ}{{\~n}}1  {Ñ}{{\~N}}1
}
