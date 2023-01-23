{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "221eb4a9",
   "metadata": {},
   "source": [
    "#1. Import the NUMPY package under the name np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f7b149d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "122ed49f",
   "metadata": {},
   "source": [
    "2. Print the NUMPY version and the configuration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "6280cba4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Name: numpyNote: you may need to restart the kernel to use updated packages.\n",
      "\n",
      "Version: 1.21.5\n",
      "Summary: NumPy is the fundamental package for array computing with Python.\n",
      "Home-page: https://www.numpy.org\n",
      "Author: Travis E. Oliphant et al.\n",
      "Author-email: \n",
      "License: BSD\n",
      "Location: c:\\users\\mocus\\anaconda3\\lib\\site-packages\n",
      "Requires: \n",
      "Required-by: astropy, bkcharts, bokeh, Bottleneck, daal4py, datashader, datashape, gensim, h5py, holoviews, hvplot, imagecodecs, imageio, matplotlib, mkl-fft, mkl-random, numba, numexpr, pandas, patsy, pyerfa, PyWavelets, scikit-image, scikit-learn, scipy, seaborn, statsmodels, tables, tifffile, xarray\n"
     ]
    }
   ],
   "source": [
    "%pip show numpy"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "382d99b8",
   "metadata": {},
   "source": [
    "3. Generate a 2x3x5 3-dimensional array with random values. Assign the array to variable \"a\"\n",
    "Challenge: there are at least three easy ways that use numpy to generate random arrays. How many ways can you find?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "0703d3f1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[0.0577469 , 0.66625389, 0.08611804, 0.90995131, 0.57561381],\n",
       "        [0.31662004, 0.20905372, 0.63589347, 0.87589039, 0.87491501],\n",
       "        [0.64260193, 0.16783597, 0.7166249 , 0.05400156, 0.40228989]],\n",
       "\n",
       "       [[0.07775553, 0.73101123, 0.75279076, 0.15545559, 0.61190491],\n",
       "        [0.11371491, 0.37091706, 0.60254253, 0.2650523 , 0.13943987],\n",
       "        [0.82661627, 0.21689146, 0.27718111, 0.63951748, 0.13445378]]])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a=np.random.random((2, 3, 5))\n",
    "\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "8805bccd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[0.66741261, 0.72567415, 0.39657854, 0.96088276, 0.64749325],\n",
       "        [0.1164518 , 0.56331638, 0.11124331, 0.96492811, 0.42551144],\n",
       "        [0.15372053, 0.3116195 , 0.61093788, 0.32748749, 0.01674448]],\n",
       "\n",
       "       [[0.89600264, 0.41849709, 0.61375865, 0.30033132, 0.14429683],\n",
       "        [0.53431597, 0.2855281 , 0.08366749, 0.03179618, 0.650213  ],\n",
       "        [0.16150221, 0.11617146, 0.04291477, 0.00906056, 0.88234729]]])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.random.rand(2, 3, 5)\n",
    "\n",
    "a"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c85111b",
   "metadata": {},
   "source": [
    "4. Print \"a\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "32f63fc9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[0.0577469  0.66625389 0.08611804 0.90995131 0.57561381]\n",
      "  [0.31662004 0.20905372 0.63589347 0.87589039 0.87491501]\n",
      "  [0.64260193 0.16783597 0.7166249  0.05400156 0.40228989]]\n",
      "\n",
      " [[0.07775553 0.73101123 0.75279076 0.15545559 0.61190491]\n",
      "  [0.11371491 0.37091706 0.60254253 0.2650523  0.13943987]\n",
      "  [0.82661627 0.21689146 0.27718111 0.63951748 0.13445378]]]\n"
     ]
    }
   ],
   "source": [
    "print(a)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35b9089c",
   "metadata": {},
   "source": [
    "#5. Create a 5x2x3 3-dimensional array with all values equaling 1.\n",
    "Assign the array to variable \"b\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "0ec33b1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "b = np.ones((5, 2, 3))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e1060f72",
   "metadata": {},
   "source": [
    "#6. Print b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3fd38c93",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]]]\n"
     ]
    }
   ],
   "source": [
    "print(b)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0fe34a35",
   "metadata": {},
   "source": [
    "7. Do a and b have the same size? How do you prove that in Python code?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "01a4f24b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#No tienen las mismas dimensiones, lo comprobamos en la siguiente celda"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "e680d3a6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 3, 5)\n",
      "(5, 2, 3)\n"
     ]
    }
   ],
   "source": [
    "print(a.shape)\n",
    "print(b.shape)   #observamos que las dimensiones no coinciden"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "410ceb1a",
   "metadata": {},
   "source": [
    "8. Are you able to add a and b? Why or why not?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "6055f312",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 3, 5)\n"
     ]
    }
   ],
   "source": [
    "#No se pueden sumar porque no tienen las mismas dimensiones"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1aa40645",
   "metadata": {},
   "source": [
    "9. Transpose b so that it has the same structure of a (i.e. become a 2x3x5 array). Assign the transposed array to varialbe \"c\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "5104d82a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "b = b = np.ones((5, 2, 3))\n",
    "\n",
    "print(b)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "1b8271d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3, 2, 5)"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "b.T.shape    # matriz.T.shape transpone la matriz \"b\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "da9a9844",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3, 2, 5)\n"
     ]
    }
   ],
   "source": [
    "c = b.T.shape     #1ª Forma de transponer\n",
    "print(c)          # he creado la matriz \"c\" al trasponer \"b\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "659142a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]]]\n"
     ]
    }
   ],
   "source": [
    "b = np.ones((5, 2, 3))               #2ª Forma de transponer\n",
    "\n",
    "c = np.transpose(b, (1, 2, 0))  \n",
    "\n",
    "print(c)                            # En este caso movemos las dimensiones a la ubicación que queremos, de ahí los números."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9282878f",
   "metadata": {},
   "source": [
    "10. Try to add a and c. Now it should work. Assign the sum to varialbe \"d\". But why does it work now?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "f89fb256",
   "metadata": {},
   "outputs": [],
   "source": [
    "d = a + c"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e98aacb7",
   "metadata": {},
   "source": [
    "#11. Print a and d. Notice the difference and relation of the two array in terms of the values? Explain."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "9956a25f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1.43678261 1.95781453 1.87639114 1.36179996 1.1359978 ]\n",
      "  [1.65683122 1.93603473 1.3534837  1.501636   1.21975027]\n",
      "  [1.36038552 1.36038367 1.786549   1.67435142 1.70594936]]\n",
      "\n",
      " [[1.32713475 1.60823889 1.20263606 1.02436121 1.9705442 ]\n",
      "  [1.48931669 1.2869412  1.95089767 1.56924245 1.76522904]\n",
      "  [1.2745515  1.28754139 1.59800115 1.82208287 1.18078185]]]\n",
      "[[[0.43678261 0.95781453 0.87639114 0.36179996 0.1359978 ]\n",
      "  [0.65683122 0.93603473 0.3534837  0.501636   0.21975027]\n",
      "  [0.36038552 0.36038367 0.786549   0.67435142 0.70594936]]\n",
      "\n",
      " [[0.32713475 0.60823889 0.20263606 0.02436121 0.9705442 ]\n",
      "  [0.48931669 0.2869412  0.95089767 0.56924245 0.76522904]\n",
      "  [0.2745515  0.28754139 0.59800115 0.82208287 0.18078185]]]\n"
     ]
    }
   ],
   "source": [
    "print(d)\n",
    "\n",
    "print(a)\n",
    "\n",
    "# le he sumado 1 a cada valor de la matriz \"d\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db83701e",
   "metadata": {},
   "source": [
    "#12. Multiply a and c. Assign the result to e"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "387072f3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]]]\n"
     ]
    }
   ],
   "source": [
    "e = a * c\n",
    "\n",
    "print(c)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "626b7857",
   "metadata": {},
   "source": [
    "#13. Does e equal to a? Why or why not?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ece044c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "2d3ff9e5",
   "metadata": {},
   "source": [
    "#14. Identify the max, min, and mean values in d. Assign those values to variables \"d_max\", \"d_min\", and \"d_mean\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "2680af33",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.9705442044647818\n"
     ]
    }
   ],
   "source": [
    "d_max = d.max()\n",
    "\n",
    "print(d_max)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "60fc4f2f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.024361212263206\n"
     ]
    }
   ],
   "source": [
    "d_min = d.min()\n",
    "\n",
    "print(d_min)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "d577f28a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.5227213942843023\n"
     ]
    }
   ],
   "source": [
    "d_mean = d.mean()\n",
    "\n",
    "print(d_mean)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9b0730b",
   "metadata": {},
   "source": [
    "#15. Now we want to label the values in d. First create an empty array \"f\" with the same shape (i.e. 2x3x5) as d using `np.empty`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "bef90c1b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]\n",
      "  [1. 1. 1. 1. 1.]]]\n"
     ]
    }
   ],
   "source": [
    "f = np.empty((2, 3, 5))\n",
    "print(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "e016d6a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1.43678261 1.95781453 1.87639114 1.36179996 1.1359978 ]\n",
      "  [1.65683122 1.93603473 1.3534837  1.501636   1.21975027]\n",
      "  [1.36038552 1.36038367 1.786549   1.67435142 1.70594936]]\n",
      "\n",
      " [[1.32713475 1.60823889 1.20263606 1.02436121 1.9705442 ]\n",
      "  [1.48931669 1.2869412  1.95089767 1.56924245 1.76522904]\n",
      "  [1.2745515  1.28754139 1.59800115 1.82208287 1.18078185]]]\n"
     ]
    }
   ],
   "source": [
    "print(d)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c52a0bbb",
   "metadata": {},
   "source": [
    "\"\"\"\n",
    "#16. Populate the values in f. For each value in d, if it's larger than d_min but smaller than d_mean, assign 25 to the corresponding value in f.\n",
    "If a value in d is larger than d_mean but smaller than d_max, assign 75 to the corresponding value in f.\n",
    "If a value equals to d_mean, assign 50 to the corresponding value in f.\n",
    "Assign 0 to the corresponding value(s) in f for d_min in d.\n",
    "Assign 100 to the corresponding value(s) in f for d_max in d.\n",
    "In the end, f should have only the following values: 0, 25, 50, 75, and 100.\n",
    "Note: you don't have to use Numpy in this question.\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "492e8323",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1.43678261 1.95781453 1.87639114 1.36179996 1.1359978 ]\n",
      "  [1.65683122 1.93603473 1.3534837  1.501636   1.21975027]\n",
      "  [1.36038552 1.36038367 1.786549   1.67435142 1.70594936]]\n",
      "\n",
      " [[1.32713475 1.60823889 1.20263606 1.02436121 1.9705442 ]\n",
      "  [1.48931669 1.2869412  1.95089767 1.56924245 1.76522904]\n",
      "  [1.2745515  1.28754139 1.59800115 1.82208287 1.18078185]]]\n",
      "[[[25. 75. 75.  1.  1.]\n",
      "  [75. 75. 25.  1.  1.]\n",
      "  [25. 25. 75.  1.  1.]]\n",
      "\n",
      " [[25. 75. 25.  1.  1.]\n",
      "  [25. 25. 75.  1.  1.]\n",
      "  [25. 25. 75.  1.  1.]]]\n"
     ]
    }
   ],
   "source": [
    "for i in range(len(d)):\n",
    "    \n",
    "    for j in range(len(d[i])):\n",
    "        \n",
    "        for k in range(len(d[i])):\n",
    "            \n",
    "            e=d[i][j][k]\n",
    "            \n",
    "            if d_min < e < d_mean:\n",
    "                \n",
    "                f[i][j][k]=25\n",
    "            \n",
    "            elif d_mean < e < d_max:\n",
    "                \n",
    "                f[i][j][k] = 75\n",
    "                \n",
    "            elif e == d_mean:\n",
    "                \n",
    "                f[i][j][k] = 50\n",
    "                \n",
    "            elif e == d_min:\n",
    "                \n",
    "                f[i][j][k]= 0\n",
    "                \n",
    "            elif e == d_max:\n",
    "                \n",
    "                f[i][j][k] = 100\n",
    "                \n",
    "print(d)\n",
    "print(f)\n",
    "                "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04feb83e",
   "metadata": {},
   "source": [
    "\"\"\"\n",
    "#17. Print d and f. Do you have your expected f?\n",
    "For instance, if your d is:\n",
    "array([[[1.85836099, 1.67064465, 1.62576044, 1.40243961, 1.88454931],\n",
    "        [1.75354326, 1.69403643, 1.36729252, 1.61415071, 1.12104981],\n",
    "        [1.72201435, 1.1862918 , 1.87078449, 1.7726778 , 1.88180042]],\n",
    "\n",
    "       [[1.44747908, 1.31673383, 1.02000951, 1.52218947, 1.97066381],\n",
    "        [1.79129243, 1.74983003, 1.96028037, 1.85166831, 1.65450881],\n",
    "        [1.18068344, 1.9587381 , 1.00656599, 1.93402165, 1.73514584]]])\n",
    "\n",
    "Your f should be:\n",
    "array([[[ 75.,  75.,  75.,  25.,  75.],\n",
    "        [ 75.,  75.,  25.,  25.,  25.],\n",
    "        [ 75.,  25.,  75.,  75.,  75.]],\n",
    "\n",
    "       [[ 25.,  25.,  25.,  25., 100.],\n",
    "        [ 75.,  75.,  75.,  75.,  75.],\n",
    "        [ 25.,  75.,   0.,  75.,  75.]]])\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "74e5c368",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1.43678261 1.95781453 1.87639114 1.36179996 1.1359978 ]\n",
      "  [1.65683122 1.93603473 1.3534837  1.501636   1.21975027]\n",
      "  [1.36038552 1.36038367 1.786549   1.67435142 1.70594936]]\n",
      "\n",
      " [[1.32713475 1.60823889 1.20263606 1.02436121 1.9705442 ]\n",
      "  [1.48931669 1.2869412  1.95089767 1.56924245 1.76522904]\n",
      "  [1.2745515  1.28754139 1.59800115 1.82208287 1.18078185]]]\n",
      "[[[25. 75. 75.  1.  1.]\n",
      "  [75. 75. 25.  1.  1.]\n",
      "  [25. 25. 75.  1.  1.]]\n",
      "\n",
      " [[25. 75. 25.  1.  1.]\n",
      "  [25. 25. 75.  1.  1.]\n",
      "  [25. 25. 75.  1.  1.]]]\n"
     ]
    }
   ],
   "source": [
    "print(d)\n",
    "print(f)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af3ee68c",
   "metadata": {},
   "source": [
    "\"\"\"\n",
    "#18. Bonus question: instead of using numbers (i.e. 0, 25, 50, 75, and 100), how to use string values \n",
    "(\"A\", \"B\", \"C\", \"D\", and \"E\") to label the array elements? You are expecting the result to be:\n",
    "array([[[ 'D',  'D',  'D',  'B',  'D'],\n",
    "        [ 'D',  'D',  'B',  'B',  'B'],\n",
    "        [ 'D',  'B',  'D',  'D',  'D']],\n",
    "\n",
    "       [[ 'B',  'B',  'B',  'B',  'E'],\n",
    "        [ 'D',  'D',  'D',  'D',  'D'],\n",
    "        [ 'B',  'D',   'A',  'D', 'D']]])\n",
    "Again, you don't need Numpy in this question.\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "254e7c72",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
