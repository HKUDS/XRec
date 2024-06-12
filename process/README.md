### XRec Data processing

**Prepare source files and save to `source` directory:**
- `review.json` contains review text data of items written by users.
- `business.json` contains full review text data including the user_id that wrote the review and the business_id the review is written for.

**The following is the order of data processing:**
1. `python interaction.py`: extract interaction information between users and items
2. `python metadata.py`: extract metadata of items
3. follow the code in `generation` directory to generate `item_profile.json`, `user_profile.json` and `explanation.json`
4. `python data.py`: generate ``data.json`` file, which contains all the data needed for the model
5. `python separate.py`: separate data into training, validation and test sets
6. `python para_dict.py`: prepare neighborhood information for collaborative filtering
