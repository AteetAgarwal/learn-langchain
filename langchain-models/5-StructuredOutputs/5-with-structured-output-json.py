from typing import Optional,Literal
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=AzureChatOpenAI(deployment_name="gpt-4o-mini")

#Review json schema
json_schema={
     "title": "review",
     "description": "schema for product review",
     "type": "object",
     "properties": {
         "key_themes": {
             "type": "array",
             "items": {
                 "type": "string"
             },
             "description": "Write down all the key themes mentioned in the review"
         },
         "summary": {
             "type": "string",
             "description": "A brief summary of the review"
         },
         "sentiment": {
             "type": "string",
             "enum": ["pos", "neg"],
             "description": "The sentiment of the review (e.g., positive, negative, neutral)"
         },
         "pros": {
             "type": "array",
             "items": {
                 "type": "string"
             },
             "description": "List the pros mentioned in the review"
         },
         "cons": {
             "type": "array",
             "items": {
                 "type": "string"
             },
             "description": "List the cons mentioned in the review"
         },
         "name": {
             "type": ["string", "null"],
             "description": "Write the name of the reviewer, if mentioned"
         }
     },
     "required": ["summary", "sentiment"]
}   

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""
I purchased this laptop three weeks ago and have been using it daily for work, casual gaming, and content creation. Overall it's been a strong performer, but there are a few notable trade-offs.

Design and Build:
The chassis is solid and feels premium for the price point. The aluminum lid resists flex and the hinge is smooth and firm, though it doesn't open with one hand. The keyboard has good travel and is comfortable for long typing sessions; the backlight is evenly lit. The trackpad is spacious and responsive, but the click mechanism is a bit stiff at the edges.

Display:
The 15.6" 1440p panel is sharp with vivid colors and excellent viewing angles. Contrast is good for a mid-range laptop, and the matte finish cuts down reflections. For color-critical work I still prefer calibrating with a hardware tool, but out of the box the display is great for photo editing and watching videos.

Performance:
Equipped with a modern multi-core CPU and a dedicated GPU, the laptop handles multitasking, video editing, and light-to-moderate gaming without breaking a sweat. Thermal management is decent — fans ramp up under load but noise levels stay acceptable. I did notice some thermal throttling under sustained heavy loads like long 4K exports, but it didn’t impede typical workflows.

Battery Life:
Battery life is good for mixed usage: I consistently get 7–8 hours of web browsing and document editing, and about 4–5 hours under heavier workloads. Fast charging is a nice addition and replenishes battery quickly when needed.

Ports and Connectivity:
The port selection is generous: multiple USB-A and USB-C ports, an HDMI output, and a full-size SD card reader which is great for photographers. Wi-Fi and Bluetooth connectivity have been reliable with no dropouts in my experience.

Software and Support:
The laptop ships with a minimal amount of bundled software, mostly useful utilities rather than bloatware. Driver updates have been regular, and customer support was responsive when I had a minor question about warranty registration.

Cons:
- Speakers are average and lack bass, so I recommend headphones for music or watching movies.
- The webcam is serviceable but not excellent in low light.
- Under sustained heavy loads there is some thermal throttling which can affect long encoding tasks.

Conclusion:
This laptop is an excellent choice for professionals and creatives who need a balance of performance, display quality, and portability without paying flagship prices. It shines in daily productivity, media editing, and light gaming, while the few minor drawbacks (speakers, webcam, and occasional thermal throttling) are manageable for most users. Overall, a highly recommended mid-range laptop with great value.
""")

print(result)