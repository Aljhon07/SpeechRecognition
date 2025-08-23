import jiwer
import pandas as pd

# All Reference–Prediction pairs
pairs = [
("he was born at wichenford in worcestershire and educated at balliol college oxford",
 "He was born in Gifford in Worcestershire and educated at a college in Oxford."),
("the portuguese division was overrun and withdrew towards estaires",
 "The Portuguese division was over, and we withdrew towards the stairs."),
("her health by this stage was also poor",
 "The hold by this stage was also poor."),
("his sporting interests outside of cricket included golf",
 "His interests outside of cricket included golf."),
("the following year he was elected to be part of the london designer collections",
 "The following year he was elected to the London Design Collections."),
("a healthy diet combined with lots of exercise can help you keep fit",
 "Diet combined with lots of exercise can help."),
("safronov is the nearest rural locality",
 "Safronov is the nearest rural locality."),
("contemporary fellow ministers in the southern baptist convention praised his preaching abilities",
 "Contemporary fellow ministers and the Southern Bantus conventionalized his preaching abilities."),
("bucknell tied for third in the colonial league",
 "Tied for third in the Colonial League."),
("the ghettoization was completed within a week",
 "The finalization was completed this week."),
("philpott was for many years treasurer and then president of the british psychological society",
 "He was a man of stature and president of the British Psychological Society."),
("both the engines and the gearbox proved to be unreliable",
 "Both the engines and props proved to be able."),
("the award went to david foster and jeremy lubbock",
 "They went to David and JB."),
("its products range from suspension forks to derailleurs",
 "Its products from suppliers for first orderers"),
("it belongs to the large family of francobelgian comics",
 "Belongs to a large family of Franco-Belgian comics."),
("he attended iowa state university where he played defense on the schools football team",
 "He attended State University, where he played defense on the school's football team."),
("some outside scholars examining the system in depth disagree with the official results",
 "Several such systemic issues without official results."),
("marshallville students attend green local school district in nearby smithville",
 "Marshfield attended the local school district in Smith High School."),
("political positions inside and outside the party are open to women",
 "Political positions inside and outside the party are open to women."),
("despite their frequent bothering he never paid them any attention",
 "Despite the frequent payments, he paid them no attention."),
("a human being cannot be safely trusted solely to the mercy of another",
 "A human being cannot be safe to trust, so later at the mercy of another."),
("one area where training specific behavior has gained significant attention is in sports",
 "One area where training specific behaviors has gained significant attention is sports."),
("however the series was launched without this technology",
 "However, the series was launched without theology."),
("wiley graduated from sault ste",
 "graduated from SOST"),
("lenny hart was also the grateful deads original money manager",
 "Hart was also the Grateful Dead's original money manager."),
("he later commented that he did not support the death of norwegian military personnel",
 "He later commented that he did not support the death of Norwegian military personnel."),
("the truth will come out one day as it happens all the time",
 "This came out one day, as it happens all of the time."),
("colonel beall served throughout the war as the only commandant of the marine corps",
 "A command to serve throughout the war as the only command of the Marine Corps."),
("it was named a notable book of the year by the new york times",
 "It was named a book of the year by The New York Times."),
("his servants obeyed his orders the monks being powerless to interfere",
 "The servants of orders are powerless from fear."),
("to deal with this unicode provides the mechanism of canonical equivalence",
 "To do this, this unique mechanism provides chemical equivalents."),
("raff was born in lachen in switzerland",
 "F was born in Lucerne, Switzerland."),
("while a large percentage consists of incoherent or chaotic sound referred to as noise",
 "While a large percentage consists of inherent or chaotic sound referred to as noise."),
("sharitylight runs in user space rather than kernel space",
 "Shitty as his rather than"),
("it stars michael gross alexis arquette and hilary swank",
 "It's my mythology, so this is a quiz?"),
("her grave is located at the hietzing cemetery",
 "Her grave is located at the Heights Cemetery."),
("future exploration will have to involve the smaller basins as well as",
 "Future explanations have involved the smaller basins as well as"),
("stevenson was born in lindsay ontario",
 "Stacy was born in St. Catharines, Ontario."),
("follow the link to find the list of world alliances",
 "For linking to find the list of analyses."),
("a teaspoon is typically characterized by its small size and long handle",
 "These are typically small and handmade."),
("these demos helped build hype around the band",
 "The SEOs helped build hype around the band."),
("once a cat burglar a master among jewel thieves",
 "Once a cat master among mountain peaks."),
("teams are seeded according to their regularseason record",
 "Insights according to the regular season data."),
("it is located about west of richfield",
 "It is located about West Ridgefield."),
("this mature wood is mainly oak and beech on clay soils",
 "The main wood is mainly oak and beech, found on soils."),
("if anything i was more competent in handling physical affairs",
 "Anything more for handling physical areas?"),
("the village is located by the deer river",
 "The village is located by the river."),
("a few tens of planets have been found around red giants",
 "A few tens of planets have been found around red giants."),
("many possibilities of what could be changed after recess were discussed",
 "Many possibilities of what could change after that were discussed."),
("his long career in radio included starring in the series dangerously yours",
 "His long long career in imaging includes Star Series Dangerous."),
("it was a time of corruption and deceitful politics",
 "It's a time of division and deceitful politics."),
("a large part of the island is naturalstate coniferous forest with some herbrich parts",
 "A large part of the island is in its natural state as forest, and some parts are ridges."),
("this gave france the race lead with great britain second",
 "This gave France the race, with Great Britain second."),
("ethel was involved with education serving on the committee of the froebel society",
 "I was involved with education, serving on the committee of the society."),
("moreover many of rikers empirical claims have been refuted",
 "Moreover, many, many of the empirical claims have been refuted."),
("i believe that compassion and empathy are crucial in fostering healthy relationships",
 "Believe compassion and empathy are crucial in most healthy relationships."),
("the stars in carina have also been found to be metalpoor",
 "The stars have also been found to have metals."),
("it seems that the evil cookies have completely replaced the lucky ones",
 "It seems that cookies have completely replaced."),
("the majority of the land of the former town is still above water",
 "The majority of the land of the former town is still above water."),
("each radial gate has a clear span",
 "Each railway gate has a clear span"),
("the group then sing the bridge and end the song repeating the chorus twice",
 "The group then singing the bridge and repeating the chorus."),
("he built the first house there",
 "He built the first house there."),
("she also portrayed forensic pathologist amy short in the francis xavier thriller poe",
 "She also portrayed pathologists M. Short and Francis X. Threlfall."),
("the filming was done with a professional crew",
 "The film was made with a professional crew."),
("the cutouts in white plaster make the facade look like lace",
 "The cuts in Whiteaster make the face like leaves."),
("walsh was born in melbourne",
 "She was born in Melbourne."),
("lawrence directed his first music videos for the san jose band a western front",
 "He directed his first music video for the San Jose band A Western Front."),
("she is married to mathieu sweeney",
 "She is married to Matt Sydney."),
("union iron works built a number of ships for the united states navy",
 "Iron Works built a number of ships for the United States Navy."),
("currently it also houses the american airpower museum",
 "Currently, it also houses the American Air Museum."),
("it can be seen from the latitude of alexandria or mobile alabama and southward",
 "It can be seen from the latitude of Alexandria or Mobile, Alabama, and south."),
("kirk refuses and the landing party is held hostage",
 "Refugees and landmines held hostage"),
("he resigned with the cubs on another minor league contract the next day",
 "He resigned with the Cubs on another minor league contract the next day."),
("followup tests can include painful biopsies which can result in excessive bleeding and infection",
 "A test can include painful biopsies, which can result in excessive infection."),
("he won the college football national championship with the florida state university seminoles",
 "He won the college football national championship with the Florida State University Seminoles.")
]

# Create case-insensitive transformation
import re
def normalize_text(text):
    # Convert to lowercase and remove punctuation, but keep spaces
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Normalize spaces
    return text.strip()

# Compute results with normalization
results = []
for i, (ref, hyp) in enumerate(pairs):
    try:
        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)
        wer = jiwer.wer(ref_norm, hyp_norm)
        results.append({
            "Reference": ref,
            "Prediction": hyp,
            "WER": round(wer, 4),
            "Score (%)": round((1-wer)*100, 2)
        })
    except Exception as e:
        print(f"Error processing pair {i+1}: {e}")

# Output as a nice table
df = pd.DataFrame(results)
print(f"\nProcessed {len(results)} out of {len(pairs)} pairs successfully")
print(df.to_string(index=False))

# Calculate average WER
if results:
    avg_wer = sum(r['WER'] for r in results) / len(results)
    avg_score = sum(r['Score (%)'] for r in results) / len(results)
    print(f"\nAverage WER: {avg_wer:.4f}")
    print(f"Average Score: {avg_score:.2f}%")

# Output as a nice table
df = pd.DataFrame(results)
print(f"\nProcessed {len(results)} out of {len(pairs)} pairs successfully")
print(df.to_string(index=False))

# Calculate average WER
if results:
    avg_wer = sum(r['WER'] for r in results) / len(results)
    avg_score = sum(r['Score (%)'] for r in results) / len(results)
    print(f"\nAverage WER: {avg_wer:.4f}")
    print(f"Average Score: {avg_score:.2f}%")
