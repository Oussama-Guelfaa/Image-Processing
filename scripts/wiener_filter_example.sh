#!/bin/bash
# Script pour tester le filtre de Wiener avec différentes valeurs de K

# Jupiter avec PSF gaussienne
echo "Test sur Jupiter avec PSF gaussienne"
imgproc damage --psf gaussian --sigma 3.0 --noise 0.01 --image data/jupiter.png --output jupiter_damaged.png

# Restauration avec différentes valeurs de K
imgproc restore --method wiener --k 0.0001 --psf gaussian --sigma 3.0 --image jupiter_damaged.png --output jupiter_restored_k0.0001.png
imgproc restore --method wiener --k 0.001 --psf gaussian --sigma 3.0 --image jupiter_damaged.png --output jupiter_restored_k0.001.png
imgproc restore --method wiener --k 0.01 --psf gaussian --sigma 3.0 --image jupiter_damaged.png --output jupiter_restored_k0.01.png
imgproc restore --method wiener --k 0.1 --psf gaussian --sigma 3.0 --image jupiter_damaged.png --output jupiter_restored_k0.1.png

# Restauration avec filtre inverse
imgproc restore --method inverse --epsilon 0.001 --psf gaussian --sigma 3.0 --image jupiter_damaged.png --output jupiter_restored_inverse.png

# Saturne avec PSF de mouvement
echo "Test sur Saturne avec PSF de mouvement"
imgproc damage --psf motion --length 15 --angle 45 --noise 0.01 --image data/saturn.png --output saturn_damaged.png

# Restauration avec différentes valeurs de K
imgproc restore --method wiener --k 0.0001 --psf motion --length 15 --angle 45 --image saturn_damaged.png --output saturn_restored_k0.0001.png
imgproc restore --method wiener --k 0.001 --psf motion --length 15 --angle 45 --image saturn_damaged.png --output saturn_restored_k0.001.png
imgproc restore --method wiener --k 0.01 --psf motion --length 15 --angle 45 --image saturn_damaged.png --output saturn_restored_k0.01.png
imgproc restore --method wiener --k 0.1 --psf motion --length 15 --angle 45 --image saturn_damaged.png --output saturn_restored_k0.1.png

# Restauration avec filtre inverse
imgproc restore --method inverse --epsilon 0.001 --psf motion --length 15 --angle 45 --image saturn_damaged.png --output saturn_restored_inverse.png

echo "Tests terminés. Vérifiez les images générées."
