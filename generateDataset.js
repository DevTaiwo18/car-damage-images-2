import fs from 'fs';
import path from 'path';
import chalk from 'chalk';

// Corrected GitHub raw URL base
const githubBaseUrl = 'https://raw.githubusercontent.com/DevTaiwo18/car-damage-images-2/main/images';

// Path to the folder containing your 46 new images
const baseImageFolder = path.join(process.cwd(), 'images');
const outputFile = path.join(process.cwd(), 'detailed-cardata.jsonl');

// Expanded questions for car damage analysis
const questions = [
    "How deep is the dent in this image?",
    "What is the recommended repair method for this damage?",
    "Is the paint affected by the dent, or is it just the body?",
    "Can this dent cause long-term issues if left unrepaired?",
    "Will this damage require replacing parts or just repair?",
    "Can this damage be fixed using paintless dent removal?",
    "How much will it cost to repair this dent?",
    "Is there any visible rust or corrosion?",
    "Is this a common type of damage for this car model?",
    "Does the paint need to be repaired after the dent is fixed?",
    "Will this damage affect the car’s resale value?"
];

// More specific responses for car damage analysis
const damageResponses = [
    "The dent in this image is about 5 cm in diameter and located on the rear door. It is shallow and suitable for paintless dent removal.",
    "This dent is moderate in size, approximately 10 cm, and is located on the front bumper. It may require filler and repainting to restore the surface.",
    "The image shows a small dent with paint scratches. Filler and repainting will be required. Estimated cost: $300 to $500.",
    "The paint is scratched, and the dent is deep. Panel replacement may be necessary for structural integrity. Estimated cost: $700 to $1,000.",
    "The dent is minor and can be fixed using paintless dent removal techniques. No significant paint damage is visible.",
    "There is visible rust around the dent, which could lead to further damage if not repaired soon. Both the rust and the dent will need to be addressed.",
    "The dent is small, but the paint has been chipped, which will require repainting after the repair. Estimated cost: $400.",
    "The image shows a deep dent on the side panel. This may impact the vehicle’s resale value if not properly repaired.",
    "The damage does not appear to affect the structural components of the car, but the paint will need repair."
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
