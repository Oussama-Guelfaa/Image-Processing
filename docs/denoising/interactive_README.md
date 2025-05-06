# Interactive Denoising in VS Code

This document explains how to run the denoising scripts interactively in VS Code, with plots displayed directly in the editor rather than in separate windows.

## Prerequisites

1. Visual Studio Code with the Python extension installed
2. Required Python packages:
   - numpy
   - matplotlib
   - scikit-image
   - matplotlib_inline (install with `pip install matplotlib-inline`)

## Running the Interactive Script

1. Open VS Code and navigate to the project directory
2. Open the `interactive_denoising.py` file
3. Right-click in the editor and select "Run Current File in Interactive Window" or use the keyboard shortcut (Shift+Enter)
4. The script will run and display the plots directly in VS Code's interactive window

## What to Expect

The interactive script will:

1. Load the `jambe.png` image from the data folder
2. Add salt and pepper noise to the image
3. Apply a median filter to remove the noise
4. Apply an adaptive median filter to remove the noise
5. Compare all denoising methods and display the results

All plots will be displayed directly in VS Code's interactive window, allowing you to see the results without switching to a separate window.

## Troubleshooting

If the plots don't appear in VS Code's interactive window:

1. Make sure you have the latest version of the Python extension for VS Code
2. Check that you have installed the `matplotlib-inline` package
3. Try restarting VS Code
4. Run the script with "Run Current File in Interactive Window" rather than "Run Python File"

## Modifying the Script

You can modify the `interactive_denoising.py` script to:

- Use different images
- Apply different noise types (gaussian, uniform, exponential)
- Change filter parameters
- Save the results to different locations

## Example Output

The script will generate plots similar to these:

1. Original Image
2. Noisy Image (Salt and Pepper)
3. Median Filtered Image
4. Adaptive Median Filtered Image
5. Comparison of all denoising methods

The comparison will also be saved to `output/interactive_denoising.png`.

## Additional Resources

- [VS Code Python Interactive Window Documentation](https://code.visualstudio.com/docs/python/jupyter-support-py)
- [Matplotlib Documentation](https://matplotlib.org/stable/users/index.html)
- [Scikit-image Documentation](https://scikit-image.org/docs/stable/)
