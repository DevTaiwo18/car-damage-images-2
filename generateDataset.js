import fs from 'fs';
import path from 'path';
import chalk from 'chalk';

// GitHub raw URL base for hosting the images
const githubBaseUrl = 'https://raw.githubusercontent.com/DevTaiwo18/car-damage-images-2/main/images';

// Path to the folder containing your new images
const baseImageFolder = path.join(process.cwd(), 'images');
const outputFile = path.join(process.cwd(), 'paliGemma-dent-dataset.jsonl');

// Questions for counting dents and assessing their size
const questions = [
    "How many dents are visible in this image?",
    "What is the size of the largest dent in this image? (Dime, Nickel, Quarter, or Half Dollar)"
];

// Responses based on dent count and size
const dentResponses = [
    "There is 1 dent visible in the image, and it is approximately the size of a Nickel.",
    "The image shows 3 dents. The largest one is the size of a Quarter, while the other two are smaller, around the size of a Dime.",
    "This image has 2 dents. Both are approximately the size of a Nickel.",
    "There are multiple dents, and the largest one appears to be around the size of a Half Dollar.",
    "There is no visible dent in this image."
];

// Function to randomly pick a dent-related response
function getAssistantResponse() {
    return dentResponses[Math.floor(Math.random() * dentResponses.length)];
}

// Function to generate JSONL content with GitHub URLs
function generateDataset() {
    fs.readdir(baseImageFolder, (err, files) => {
        if (err) {
            console.error(chalk.red('Error reading image folder:'), err);
            return;
        }

        const jsonlEntries = [];

        files.forEach((file, index) => {
            const imagePathOnGitHub = `${githubBaseUrl}/${encodeURIComponent(file)}`;

            // Add a JSONL entry for each image with GitHub URL and a focused question
            jsonlEntries.push(JSON.stringify({
                messages: [
                    {
                        role: "system",
                        content: "You are an AI that specializes in analyzing car dents, counting them, and assessing their size based on predefined categories (Dime, Nickel, Quarter, Half Dollar)."
                    },
                    {
                        role: "user",
                        content: [
                            {
                                type: "image_url",
                                image_url: {
                                    url: imagePathOnGitHub
                                }
                            },
                            {
                                type: "text",
                                text: `${questions[Math.floor(Math.random() * questions.length)]}`
                            }
                        ]
                    },
                    {
                        role: "assistant",
                        content: getAssistantResponse()
                    }
                ]
            }));
        });

        // Write the entries to the JSONL file
        fs.writeFileSync(outputFile, jsonlEntries.join('\n'));
        console.log(chalk.green(`Dataset has been written to ${outputFile}`));
    });
}

// Run the function to generate the dataset
generateDataset();
