import fs from 'fs';
import path from 'path';
import chalk from 'chalk';

// Corrected GitHub raw URL base
const githubBaseUrl = 'https://raw.githubusercontent.com/DevTaiwo18/car-damage-images-2/main/images';

// Path to the folder containing your 46 new images
const baseImageFolder = path.join(process.cwd(), 'car-damage-images-2');
const outputFile = path.join(process.cwd(), 'new-cardata.jsonl');

// Detailed questions for car damage analysis
const questions = [
    "How deep is the dent in this image?",
    "What is the recommended repair method for this damage?",
    "Is the paint affected by the dent, or is it just the body?",
    "Can this dent cause long-term issues if left unrepaired?",
    "Will this damage require replacing parts or just repair?",
    "Can this damage be fixed using paintless dent removal?",
    "How much will it cost to repair this dent?",
    "How severe is this dent compared to typical damage?"
];

// More specific responses for car damage analysis
const damageResponses = [
    "The dent in this image appears to be shallow and can likely be repaired using paintless dent removal.",
    "The dent is moderate in size, and it may require filler and repainting to restore the surface.",
    "This damage is cosmetic and does not appear to affect the vehicle's performance, but repainting may be required.",
    "The dent is deep, and replacement of the affected panel may be necessary.",
    "The paint has been scratched, and it will need to be repainted after the dent is repaired.",
    "The dent looks minor, and it can be fixed easily without affecting the car's structure.",
    "This dent may cause rust or corrosion over time if not repaired, but it is not structurally damaging.",
    "The dent is small enough to be fixed using basic paintless dent repair techniques, with minimal cost."
];

// Function to randomly pick a damage-related response
function getAssistantResponse() {
    return damageResponses[Math.floor(Math.random() * damageResponses.length)];
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
                        content: "You are an AI that analyzes car damage and provides detailed damage reports."
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
