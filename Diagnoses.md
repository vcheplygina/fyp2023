# What types of diagnoses are there, and how do they relate to each
# other? You can use Google Scholar or other sources, to learn more
# background about these diagnoses.

ACK: Actinic Keratosis
BCC: Basal Cell Carcinoma
MEL: Melanoma
NEV: Nevus
SCC: Squamous Cell Carcinoma
SEK: Seborrheic Keratosis

Actinic Keratosis (ACK) is a precancerous skin condition caused by long-term exposure to the sun. Basal Cell Carcinoma (BCC) and Squamous Cell Carcinoma (SCC) are both types of skin cancer, with BCC being the most common type. Melanoma (MEL) is a type of skin cancer that can spread to other parts of the body and is the deadliest form of skin cancer. Nevus (NEV) refers to a pigmented lesion on the skin, commonly known as a mole. Seborrheic Keratosis (SEK) is a benign skin growth that often appears as a waxy or wart-like lesion.
All BCC, SCC, and MEL are biopsy-proven. The remaining ones may have clinical diagnosis according to a consensus of a group of dermatologists. In total, approximately 58% of the samples in this dataset are biopsy-proven. This information is described in the metadata.


# Is there some missing data? Are there images of low quality? Etc

From the metadata one can see, that some cells are empty. We can't say why this happend, but we can assume that people didn't want to give permission on gathering of a specific data.


# You can select a small (e.g. 100) subset images to focus on at the start. The only requirement is that there are at least two categories of images.


# Assymetry and Color 

- `Melanoma`:
    Six suspicious shades of color: white, red, light brown, dark brown, blue-gray, and black.
    larger than 6mm

- 'Actinic Keratosis':
    Color variations, including pink, red or brown. Tender or asymptomatic White or yellow; scaly, warty, or horny surface. Skin coloured, red, or pigmented. Rough, dry or scaly patch of skin, usually less than 1 inch (2.5 centimeters) in diameter.

- 'Basal cell carcinoma':
  Generally considered to be asymmetric in its growth and appearance. Unlike benign moles or lesions, which often exhibit symmetry, BCC lesions typically have irregular or asymmetric shapes. Pearly or translucent: BCC lesions often have a pearly or translucent appearance. They may appear shiny or have a glass-like quality to them. Pink or flesh-colored: BCC lesions can have a pinkish hue or blend in with the surrounding skin color. They may resemble a flesh-colored bump or patch on the skin. Red or irritated: In some cases, BCC may appear red or irritated, especially if the tumor has ulcerated or developed into an open sore.

[asymmetric, pink, flesh-colored, red]


- 'Nevus': Nevus, also commonly known as a mole, can exhibit a range of characteristics, including symmetry or asymmetry. The symmetry of a nevus can vary from person to person and even within an individual's moles. Some moles may appear symmetric, where both halves are relatively equal in shape and size. These types of moles are often referred to as symmetric or regular moles. However, it's important to note that not all moles are symmetric. Many moles exhibit some degree of asymmetry, where one half of the mole differs from the other half in terms of shape, size, or color. Brown: Many nevi are brown in color, ranging from light tan to dark brown. This is the most typical color for moles. Black: Some moles may appear black, especially those that contain a higher concentration of pigment. Flesh-colored: Certain nevi can be flesh-colored or slightly pink. These types of moles may blend in with the surrounding skin. Red: In rare cases, nevi may appear red or reddish-pink. These are typically referred to as vascular or cherry angioma-like moles and are caused by an increase in blood vessels. Blue: Blue-colored nevi are less common but can occur. These moles appear blue or bluish-gray due to the way light scatters within the skin.

[Symmetric but can be assymetric. Brown, black, red, flesh-colored, blue.]


- 'Squamous Cell Carcinoma': Reddish or pinkish: SCC lesions often appear reddish or pinkish in color. They may have a slightly raised or rough surface and can resemble a persistent sore or ulcer. Crusty or scaly: Some SCCs develop a crust or scale on the surface, which can range in color from yellowish to brownish. The presence of crust or scales may be indicative of a more advanced or aggressive SCC. Flesh-colored: In some cases, especially in the early stages, squamous cell carcinomas can be flesh-colored. They may blend in with the surrounding skin and appear as a firm, flesh-colored bump. In general, SCC lesions may be more likely to have irregular or asymmetric shapes compared to benign moles or lesions. They may have uneven borders and varying sizes within the same lesion. 

[Asymmetric, Red, pink, flesh-colored, yellow, brown]


- 'Seborrheic Keratosis': Tan or light brown: Most seborrheic keratoses appear tan or light brown in color. They may resemble a stuck-on or waxy lesion on the skin. Dark brown or black: In some cases, seborrheic keratoses can have a darker brown or black color. This is more common in larger or thicker growths. Yellowish or whitish: Some seborrheic keratoses may have a yellowish or whitish coloration. This is more apparent in lesions that have a thicker, greasy, or keratin-filled appearance. Variegated: Seborrheic keratoses can sometimes have multiple colors within the same lesion. These variations can include a combination of tan, brown, black, yellow, or white patches. Seborrheic keratosis (SK) lesions are typically symmetric or evenly balanced in shape and appearance. Unlike certain types of skin cancer that may exhibit asymmetry, seborrheic keratoses often have a symmetrical growth pattern.

[Symmetric, tan, white, yellow, black, brown]