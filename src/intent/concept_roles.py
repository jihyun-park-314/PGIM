"""
concept_roles.py
----------------
Lightweight concept-role taxonomy for Amazon Movies & TV.

Each category concept is assigned a role. Only goal-eligible roles are used
as goal_concepts for short-term intent. Non-semantic roles are excluded from
exploration goal selection (and optionally from persona signal).

Role taxonomy:
  STRONG_SEMANTIC  - named content genre/subgenre/content-type (drama, horror, action,
                     animation, documentary, sci-fi, western, romance, comedy, ...)
                     These are the PRIMARY candidates for exploration goal_concepts.
  WEAK_DESCRIPTOR  - mood/tone/adjective-like tags (bleak, dark, cerebral, atmospheric,
                     haunting, gritty, futuristic, emotional, whimsical, ominous, ...)
                     These are AUXILIARY evidence; not primary goal direction.
  PLATFORM         - platform/source anchor (prime_video)
  NAVIGATION       - storefront/browse structure (featured_categories, all_titles, ...)
  PROMO_DEAL       - promotional/deal sections (studio_specials, today's_deals, ...)
  PUBLISHER        - studio/distributor label (warner_home_video, sony_pictures_..., ...)
  FORMAT_META      - physical format metadata (blu-ray, dvd, dts, widescreen, ...)
  FRANCHISE        - named fictional franchise/universe (star_wars, harry_potter, x-men, ...)
                     SemanticAnchor zone; goal-eligible for task_focus context
  EDITION          - editorial/physical curated edition (criterion_collection, box_sets, ...)
                     ProductContext zone; goal-eligible for task_focus context
  COLLECTION       - publisher-catalog dumps and storefront editorial tags (all_*_titles, ...)
                     NoiseMeta zone; never goal-eligible
  UMBRELLA         - top-level umbrella too broad to be useful (movies_&_tv, movies, tv, ...)
  PERSON           - named person/band/artist
  TEMPORAL         - time-period or seasonal (christmas, the_1980s, ...)
  GEO_LANG         - country/language (france, italian, foreign_films, ...)
  AGE_DEMO         - age/demographic (kids_&_family, tweens, ...)
  AWARD            - award labels (oscar_nominees, emmy_central, ...)
  NETWORK_CHANNEL  - TV network/channel (bbc, fox_tv, discovery_channel, ...)

Usage:
  from src.intent.concept_roles import is_goal_eligible, is_strong_semantic, CONCEPT_ROLES
"""

from __future__ import annotations

# ── Role assignments for all 635 category concepts ────────────────────────────
# Only `category:` prefix concepts are present here.
# Unknown concepts (not in this dict) default to STRONG_SEMANTIC to be conservative.

CONCEPT_ROLES: dict[str, str] = {
    "category:10-12_years": "AGE_DEMO",
    "category:20/20": "NETWORK_CHANNEL",
    "category:20th_century_fox_home_entertainment": "PUBLISHER",
    "category:3-6_years": "AGE_DEMO",
    "category:4-for-3_dvd": "PROMO_DEAL",
    "category:60_minutes_store": "NETWORK_CHANNEL",
    "category:7-11_years": "AGE_DEMO",
    "category:7-9_years": "AGE_DEMO",
    "category:a&e_home_video": "PUBLISHER",
    "category:a&e_original_movies": "NETWORK_CHANNEL",
    "category:abba": "PERSON",
    "category:abc_news": "NETWORK_CHANNEL",
    "category:abc_news_classics": "NETWORK_CHANNEL",
    "category:abc_tv_shows": "NETWORK_CHANNEL",
    "category:ac/dc": "PERSON",
    "category:acorn_deals": "PROMO_DEAL",
    "category:acting_troupes_&_companies": "STRONG_SEMANTIC",
    "category:action": "STRONG_SEMANTIC",
    "category:action_&_adventure": "STRONG_SEMANTIC",
    "category:adventure": "STRONG_SEMANTIC",
    "category:adventures": "STRONG_SEMANTIC",
    "category:aerosmith": "PERSON",
    "category:africa": "GEO_LANG",
    "category:african_american_cinema": "STRONG_SEMANTIC",
    "category:african_american_heritage": "STRONG_SEMANTIC",
    "category:aguilera,_christina": "PERSON",
    "category:alien_invasion": "STRONG_SEMANTIC",
    "category:alien_saga": "FRANCHISE",
    "category:alienated": "WEAK_DESCRIPTOR",
    "category:aliens": "STRONG_SEMANTIC",
    "category:all": "NAVIGATION",
    "category:all_a&e_titles": "COLLECTION",
    "category:all_bbc_titles": "COLLECTION",
    "category:all_beauty": "NAVIGATION",
    "category:all_disney_titles": "COLLECTION",
    "category:all_electronics": "NAVIGATION",
    "category:all_fox_titles": "COLLECTION",
    "category:all_fx_shows": "COLLECTION",
    "category:all_hbo_titles": "COLLECTION",
    "category:all_lionsgate_titles": "COLLECTION",
    "category:all_made-for-tv_movies": "COLLECTION",
    "category:all_mgm_titles": "COLLECTION",
    "category:all_mtv": "COLLECTION",
    "category:all_new_yorker_titles": "COLLECTION",
    "category:all_sci_fi_channel_shows": "COLLECTION",
    "category:all_showtime_titles": "COLLECTION",
    "category:all_sony_pictures_classics": "COLLECTION",
    "category:all_sony_pictures_titles": "COLLECTION",
    "category:all_sundance_titles": "COLLECTION",
    "category:all_terminator": "FRANCHISE",
    "category:all_titles": "NAVIGATION",
    "category:all_universal_studios_titles": "COLLECTION",
    "category:amazon_fashion": "NAVIGATION",
    "category:amazon_home": "NAVIGATION",
    "category:ambitious": "WEAK_DESCRIPTOR",
    "category:american_film_institute_series": "EDITION",
    "category:anchor_bay_deals": "PROMO_DEAL",
    "category:anchor_bay_horror_store": "PUBLISHER",
    "category:animal_planet": "NETWORK_CHANNEL",
    "category:animated_cartoons": "STRONG_SEMANTIC",
    "category:animated_characters": "STRONG_SEMANTIC",
    "category:animated_movies": "STRONG_SEMANTIC",
    "category:animation": "STRONG_SEMANTIC",
    "category:anime": "STRONG_SEMANTIC",
    "category:anime_&_manga": "STRONG_SEMANTIC",
    "category:anxious": "WEAK_DESCRIPTOR",
    "category:armstrong,_louis": "PERSON",
    "category:art_&_artists": "STRONG_SEMANTIC",
    "category:art_house_&_international": "STRONG_SEMANTIC",
    "category:arthouse": "STRONG_SEMANTIC",
    "category:arts,_entertainment,_and_culture": "STRONG_SEMANTIC",
    "category:arts_&_entertainment": "STRONG_SEMANTIC",
    "category:atmospheric": "WEAK_DESCRIPTOR",
    "category:australia_&_new_zealand": "GEO_LANG",
    "category:award_winners_in_movies_&_tv": "AWARD",
    "category:backstreet_boys": "PERSON",
    "category:ballet_&_dance": "STRONG_SEMANTIC",
    "category:basie,_count": "PERSON",
    "category:batman_animated_movies": "FRANCHISE",
    "category:bbc": "NETWORK_CHANNEL",
    "category:beastie_boys": "PERSON",
    "category:beautiful": "WEAK_DESCRIPTOR",
    "category:bee_gees": "PERSON",
    "category:best_of_2013": "TEMPORAL",
    "category:bible": "STRONG_SEMANTIC",
    "category:biographies": "STRONG_SEMANTIC",
    "category:biography": "STRONG_SEMANTIC",
    "category:birth-2_years": "AGE_DEMO",
    "category:biting": "WEAK_DESCRIPTOR",
    "category:blakey,_art": "PERSON",
    "category:blaxploitation": "STRONG_SEMANTIC",
    "category:bleak": "WEAK_DESCRIPTOR",
    "category:blu-ray": "FORMAT_META",
    "category:bold": "WEAK_DESCRIPTOR",
    "category:bon_jovi": "PERSON",
    "category:books": "NAVIGATION",
    "category:borge,_victor": "PERSON",
    "category:bowie,_david": "PERSON",
    "category:box_sets": "EDITION",
    "category:boxed_sets": "EDITION",
    "category:brazil": "GEO_LANG",
    "category:breakthrough_cinema": "STRONG_SEMANTIC",
    "category:british_cult_television": "STRONG_SEMANTIC",
    "category:british_television": "GEO_LANG",
    "category:broadway": "STRONG_SEMANTIC",
    "category:broadway_theatre_archive": "EDITION",
    "category:brooks,_garth": "PERSON",
    "category:bryan_kest": "PERSON",
    "category:buena_vista_social_club": "PERSON",
    "category:by_age": "NAVIGATION",
    "category:by_animator": "NAVIGATION",
    "category:by_country": "NAVIGATION",
    "category:by_genre": "NAVIGATION",
    "category:by_instructor": "NAVIGATION",
    "category:by_original_language": "NAVIGATION",
    "category:camp": "WEAK_DESCRIPTOR",
    "category:campy": "WEAK_DESCRIPTOR",
    "category:canada": "GEO_LANG",
    "category:carey,_mariah": "PERSON",
    "category:carpenters": "PERSON",
    "category:cartoon_network": "NETWORK_CHANNEL",
    "category:cash,_johnny": "PERSON",
    "category:cbs_news_network": "NETWORK_CHANNEL",
    "category:cell_phones_&_accessories": "NAVIGATION",
    "category:celtic_woman": "PERSON",
    "category:cerebral": "WEAK_DESCRIPTOR",
    "category:channels": "NAVIGATION",
    "category:characters_&_series": "NAVIGATION",
    "category:charming": "WEAK_DESCRIPTOR",
    "category:cheerful": "WEAK_DESCRIPTOR",
    "category:china": "GEO_LANG",
    "category:christian_movies_&_tv": "STRONG_SEMANTIC",
    "category:christian_video": "STRONG_SEMANTIC",
    "category:christmas": "TEMPORAL",
    "category:ciencia_ficción_y_fantasía": "STRONG_SEMANTIC",
    "category:cirque_du_soleil": "PERSON",
    "category:clapton,_eric": "PERSON",
    "category:classic_tv": "STRONG_SEMANTIC",
    "category:classical": "STRONG_SEMANTIC",
    "category:classics": "STRONG_SEMANTIC",
    "category:classics_kids_love": "COLLECTION",
    "category:coarse": "WEAK_DESCRIPTOR",
    "category:collections": "NAVIGATION",
    "category:collections_&_documentaries": "NAVIGATION",
    "category:collins,_phil": "PERSON",
    "category:coltrane,_john": "PERSON",
    "category:comedy": "STRONG_SEMANTIC",
    "category:comedy_central_presents": "NETWORK_CHANNEL",
    "category:compelling": "WEAK_DESCRIPTOR",
    "category:computer_animation": "STRONG_SEMANTIC",
    "category:computers": "NAVIGATION",
    "category:confused": "WEAK_DESCRIPTOR",
    "category:contemplative": "WEAK_DESCRIPTOR",
    "category:cooper,_alice": "PERSON",
    "category:cops_&_triads": "STRONG_SEMANTIC",
    "category:cream": "PERSON",
    "category:crime_&_conspiracy": "STRONG_SEMANTIC",
    "category:criterion_collection": "EDITION",
    "category:critic's_choice": "COLLECTION",  # editorial tag → NoiseMeta
    "category:crow,_sheryl": "PERSON",
    "category:cult_classics": "STRONG_SEMANTIC",
    "category:cult_movies": "STRONG_SEMANTIC",
    "category:dance": "STRONG_SEMANTIC",
    "category:dance_&_music": "STRONG_SEMANTIC",
    "category:dark": "WEAK_DESCRIPTOR",
    "category:davis,_miles": "PERSON",
    "category:dc_comics_collection": "FRANCHISE",
    "category:dc_universe_animated_original_movies": "FRANCHISE",
    "category:deep_purple": "PERSON",
    "category:denver,_john": "PERSON",
    "category:digital_music": "FORMAT_META",
    "category:digital_vhs": "FORMAT_META",
    "category:disc_on_demand": "FORMAT_META",
    "category:discovery_channel": "NETWORK_CHANNEL",
    "category:disney_channel": "NETWORK_CHANNEL",
    "category:disney_channel_original_movies": "NETWORK_CHANNEL",
    "category:disney_channel_series": "NETWORK_CHANNEL",
    "category:disney_home_video": "PUBLISHER",
    "category:documentary": "STRONG_SEMANTIC",
    "category:docurama": "PUBLISHER",
    "category:dove-approved_family_films": "COLLECTION",  # editorial approval tag → NoiseMeta
    "category:dove_approved_list": "COLLECTION",           # editorial approval tag → NoiseMeta
    "category:downbeat": "WEAK_DESCRIPTOR",
    "category:dr._dre": "PERSON",
    "category:drama": "STRONG_SEMANTIC",
    "category:dreamlike": "WEAK_DESCRIPTOR",
    "category:dreamworks": "PUBLISHER",
    "category:dts": "FORMAT_META",
    "category:dune": "FRANCHISE",
    "category:dvd": "FORMAT_META",
    "category:dylan,_bob": "PERSON",
    "category:easter": "TEMPORAL",
    "category:easter_movies_&_tv_shows": "TEMPORAL",
    "category:easygoing": "WEAK_DESCRIPTOR",
    "category:edifying": "WEAK_DESCRIPTOR",
    "category:editor's_picks": "COLLECTION",    # editorial pick tag → NoiseMeta
    "category:educational": "STRONG_SEMANTIC",
    "category:eerie": "WEAK_DESCRIPTOR",
    "category:electrifying": "WEAK_DESCRIPTOR",
    "category:ellington,_duke": "PERSON",
    "category:emerson,_lake_&_palmer": "PERSON",
    "category:emmy_central": "AWARD",
    "category:emmy_nominees": "AWARD",
    "category:emotional": "WEAK_DESCRIPTOR",
    "category:enigmatic": "WEAK_DESCRIPTOR",
    "category:essential_art_house": "EDITION",
    "category:exciting": "WEAK_DESCRIPTOR",
    "category:exercise_&_fitness": "STRONG_SEMANTIC",
    "category:exotic": "WEAK_DESCRIPTOR",
    "category:exploitation": "STRONG_SEMANTIC",
    "category:explore_50_years_of_great_tv": "COLLECTION",
    "category:extended_editions": "FORMAT_META",
    "category:faith_&_spirituality": "STRONG_SEMANTIC",
    "category:faith_and_spirituality": "STRONG_SEMANTIC",
    "category:family_features": "STRONG_SEMANTIC",
    "category:family_friendly": "STRONG_SEMANTIC",
    "category:fantastic": "WEAK_DESCRIPTOR",
    "category:fantasy": "STRONG_SEMANTIC",
    "category:father's_day": "TEMPORAL",
    "category:feature_films": "STRONG_SEMANTIC",
    "category:featured_categories": "NAVIGATION",
    "category:featured_deals_&_new_releases": "PROMO_DEAL",
    "category:feel-good": "WEAK_DESCRIPTOR",
    "category:film_history_&_film_making": "STRONG_SEMANTIC",
    "category:film_movement": "PUBLISHER",
    "category:first_to_know": "NAVIGATION",
    "category:fitness": "STRONG_SEMANTIC",
    "category:fitness_&_yoga": "STRONG_SEMANTIC",
    "category:fleetwood_mac": "PERSON",
    "category:focus_features": "PUBLISHER",
    "category:fogerty,_john": "PERSON",
    "category:for_the_whole_family": "AGE_DEMO",
    "category:foreign_films": "GEO_LANG",
    "category:foreign_spotlight": "GEO_LANG",
    "category:fox_featured_deals": "PROMO_DEAL",
    "category:fox_news": "NETWORK_CHANNEL",
    "category:fox_tv": "NETWORK_CHANNEL",
    "category:france": "GEO_LANG",
    "category:french": "GEO_LANG",
    "category:french_new_wave": "STRONG_SEMANTIC",
    "category:frightening": "WEAK_DESCRIPTOR",
    "category:full_moon_video": "PUBLISHER",
    "category:fully_loaded_dvds": "FORMAT_META",
    "category:fun": "WEAK_DESCRIPTOR",
    "category:futuristic": "WEAK_DESCRIPTOR",
    "category:fx": "NETWORK_CHANNEL",
    "category:fx_network": "NETWORK_CHANNEL",
    "category:gaiam": "PUBLISHER",
    "category:general": "NAVIGATION",
    "category:genesis": "PERSON",
    "category:genre_for_featured_categories": "NAVIGATION",
    "category:gentle": "WEAK_DESCRIPTOR",
    "category:george_harrison": "PERSON",
    "category:german": "GEO_LANG",
    "category:germany": "GEO_LANG",
    "category:gerry_anderson": "PERSON",
    "category:gift_ideas_in_movies_&_tv": "NAVIGATION",
    "category:gillespie,_dizzy": "PERSON",
    "category:godzilla": "FRANCHISE",
    "category:golden_globe_central": "AWARD",
    "category:golden_globe_nominees": "AWARD",
    "category:golden_globe_winners": "AWARD",
    "category:grateful_dead": "PERSON",
    "category:gritty": "WEAK_DESCRIPTOR",
    "category:haggard,_merle": "PERSON",
    "category:hallmark_home_video": "PUBLISHER",
    "category:halloween_movies_&_tv_shows": "TEMPORAL",
    "category:hanna-barbera": "PUBLISHER",
    "category:harrowing": "WEAK_DESCRIPTOR",
    "category:harry_potter": "FRANCHISE",
    "category:harry_potter_and_the_chamber_of_secrets": "FRANCHISE",
    "category:harry_potter_and_the_deathly_hallows": "FRANCHISE",
    "category:harry_potter_and_the_goblet_of_fire": "FRANCHISE",
    "category:harry_potter_and_the_half-blood_prince": "FRANCHISE",
    "category:harry_potter_and_the_order_of_the_phoenix": "FRANCHISE",
    "category:harry_potter_and_the_prisoner_of_azkaban": "FRANCHISE",
    "category:harry_potter_and_the_sorcerer's_stone": "FRANCHISE",
    "category:haunted_history": "STRONG_SEMANTIC",
    "category:haunting": "WEAK_DESCRIPTOR",
    "category:hbo": "NETWORK_CHANNEL",
    "category:hbo_deals": "PROMO_DEAL",
    "category:health": "STRONG_SEMANTIC",
    "category:health_&_personal_care": "NAVIGATION",
    "category:heartwarming": "WEAK_DESCRIPTOR",
    "category:hendrix,_jimi": "PERSON",
    "category:hindi": "GEO_LANG",
    "category:historical": "STRONG_SEMANTIC",
    "category:historical_context": "STRONG_SEMANTIC",
    "category:history": "STRONG_SEMANTIC",
    "category:history_channel": "NETWORK_CHANNEL",
    "category:holiday,_billie": "PERSON",
    "category:holidays_&_seasonal": "TEMPORAL",
    "category:hollywood_vault": "EDITION",
    "category:holocaust": "STRONG_SEMANTIC",
    "category:hong_kong": "GEO_LANG",
    "category:hong_kong_action": "STRONG_SEMANTIC",
    "category:horror": "STRONG_SEMANTIC",
    "category:hungarian": "GEO_LANG",
    "category:hungary": "GEO_LANG",
    "category:iceland": "GEO_LANG",
    "category:il_divo": "PERSON",
    "category:imaginative": "WEAK_DESCRIPTOR",
    "category:independently_distributed": "PUBLISHER",
    "category:indie_&_art_house": "STRONG_SEMANTIC",
    "category:industrial_&_scientific": "STRONG_SEMANTIC",
    "category:infinifilm_edition": "FORMAT_META",
    "category:inspirational": "WEAK_DESCRIPTOR",
    "category:inspiring": "WEAK_DESCRIPTOR",
    "category:intense": "WEAK_DESCRIPTOR",
    "category:international": "GEO_LANG",
    "category:introspective": "WEAK_DESCRIPTOR",
    "category:iran": "GEO_LANG",
    "category:ireland": "GEO_LANG",
    "category:iron_maiden": "PERSON",
    "category:italian": "GEO_LANG",
    "category:italy": "GEO_LANG",
    "category:jackass": "FRANCHISE",
    "category:jackson,_janet": "PERSON",
    "category:jackson,_michael": "PERSON",
    "category:james_bond": "FRANCHISE",
    "category:jane_austen_on_dvd_store": "EDITION",
    "category:japan": "GEO_LANG",
    "category:japanese": "GEO_LANG",
    "category:jarrett,_keith": "PERSON",
    "category:jesus": "STRONG_SEMANTIC",
    "category:jewish_heritage": "STRONG_SEMANTIC",
    "category:joan_of_arc": "STRONG_SEMANTIC",
    "category:joel,_billy": "PERSON",
    "category:john,_elton": "PERSON",
    "category:john_lennon": "PERSON",
    "category:jones,_george": "PERSON",
    "category:joyous": "WEAK_DESCRIPTOR",
    "category:justice_league_animated_movies": "FRANCHISE",
    "category:kids": "AGE_DEMO",
    "category:kids_&_family": "AGE_DEMO",
    "category:king,_b.b.": "PERSON",
    "category:king,_carole": "PERSON",
    "category:kiss": "PERSON",
    "category:landmark_cult_classics": "EDITION",
    "category:lgbt": "STRONG_SEMANTIC",
    "category:lgbtq": "STRONG_SEMANTIC",
    "category:lifetime_original_movies": "NETWORK_CHANNEL",
    "category:lions_gate_home_entertainment": "PUBLISHER",
    "category:lionsgate_dvds_under_$10": "PROMO_DEAL",
    "category:lionsgate_dvds_under_$15": "PROMO_DEAL",
    "category:lionsgate_dvds_under_$20": "PROMO_DEAL",
    "category:lionsgate_home_entertainment": "PUBLISHER",
    "category:lionsgate_indie_selects": "EDITION",
    "category:live_action": "STRONG_SEMANTIC",
    "category:made-for-tv_movies": "STRONG_SEMANTIC",
    "category:madonna": "PERSON",
    "category:major_league_baseball": "STRONG_SEMANTIC",
    "category:malicious": "WEAK_DESCRIPTOR",
    "category:mamoru_oshii": "PERSON",
    "category:manilow,_barry": "PERSON",
    "category:mary-kate_&_ashley": "PERSON",
    "category:mayer,_john": "PERSON",
    "category:mclachlan,_sarah": "PERSON",
    "category:megadeth": "PERSON",
    "category:men_in_black": "FRANCHISE",
    "category:metallica": "PERSON",
    "category:mexico": "GEO_LANG",
    "category:mgm_avant-garde": "EDITION",
    "category:mgm_contemporary_classics": "EDITION",
    "category:mgm_dvds_under_$15": "PROMO_DEAL",
    "category:mgm_family_entertainment": "EDITION",
    "category:mgm_home_entertainment": "PUBLISHER",
    "category:mgm_midnite_movies": "EDITION",
    "category:mgm_movie_time": "EDITION",
    "category:mgm_musicals": "EDITION",
    "category:mgm_screen_epics": "EDITION",
    "category:mgm_vintage_classics": "EDITION",
    "category:mgm_western_legends": "EDITION",
    "category:military_&_war": "STRONG_SEMANTIC",
    "category:military_and_war": "STRONG_SEMANTIC",
    "category:mingus,_charles": "PERSON",
    "category:mini-dvd": "FORMAT_META",
    "category:miniseries": "STRONG_SEMANTIC",
    "category:miramax_home_entertainment": "PUBLISHER",
    "category:miramax_home_video": "PUBLISHER",
    "category:mod_createspace_video": "PUBLISHER",
    "category:modern_adaptations": "STRONG_SEMANTIC",
    "category:monster_movies": "STRONG_SEMANTIC",
    "category:monsters_&_mutants": "STRONG_SEMANTIC",
    "category:monthly_deals": "PROMO_DEAL",
    "category:monty_python_store": "FRANCHISE",
    "category:more_to_explore": "NAVIGATION",
    "category:morrissey_&_the_smiths": "PERSON",
    "category:mother's_day": "TEMPORAL",
    "category:mother's_day_gift_ideas_in_movies_&_tv": "TEMPORAL",
    "category:motley_crue": "PERSON",
    "category:movies": "UMBRELLA",
    "category:movies_&_tv": "UMBRELLA",
    "category:movies_&_tv_halloween_store": "TEMPORAL",
    "category:movies_&_tv_holiday_wishlist": "TEMPORAL",
    "category:mtv": "NETWORK_CHANNEL",
    "category:music_&_musicals": "STRONG_SEMANTIC",
    "category:music_&_performing_arts": "STRONG_SEMANTIC",
    "category:music_artists": "NAVIGATION",
    "category:music_video_&_concerts": "STRONG_SEMANTIC",
    "category:music_videos_&_concerts": "STRONG_SEMANTIC",
    "category:music_videos_and_concerts": "STRONG_SEMANTIC",
    "category:musicals": "STRONG_SEMANTIC",
    "category:musicals_&_performing_arts": "STRONG_SEMANTIC",
    "category:musicals_&_performing_arts_movies": "STRONG_SEMANTIC",
    "category:mysterious": "WEAK_DESCRIPTOR",
    "category:mystery": "STRONG_SEMANTIC",
    "category:mystery_&_suspense": "STRONG_SEMANTIC",
    "category:mystery_&_thrillers": "STRONG_SEMANTIC",
    "category:mystical": "WEAK_DESCRIPTOR",
    "category:n_sync": "PERSON",
    "category:nail-biting": "WEAK_DESCRIPTOR",
    "category:nature_&_wildlife": "STRONG_SEMANTIC",
    "category:nelson,_willie": "PERSON",
    "category:new_line_platinum_series": "EDITION",
    "category:new_yorker_films": "PUBLISHER",
    "category:niche_picks": "COLLECTION",   # editorial pick tag → NoiseMeta
    "category:nightline": "NETWORK_CHANNEL",
    "category:noah": "STRONG_SEMANTIC",
    "category:nostalgic": "WEAK_DESCRIPTOR",
    "category:nova": "NETWORK_CHANNEL",
    "category:office_products": "NAVIGATION",
    "category:olympics": "STRONG_SEMANTIC",
    "category:ominous": "WEAK_DESCRIPTOR",
    "category:opera": "STRONG_SEMANTIC",
    "category:optimistic": "WEAK_DESCRIPTOR",
    "category:osbourne,_ozzy": "PERSON",
    "category:oscar_central": "AWARD",
    "category:oscar_nominees": "AWARD",
    "category:oscar®_collection": "AWARD",
    "category:other_instructors": "NAVIGATION",
    "category:outlandish": "WEAK_DESCRIPTOR",
    "category:outrageous": "WEAK_DESCRIPTOR",
    "category:paramount_home_entertainment": "PUBLISHER",
    "category:parker,_charlie": "PERSON",
    "category:passionate": "WEAK_DESCRIPTOR",
    "category:passport_to_europe": "GEO_LANG",
    "category:past_emmy_winners": "AWARD",
    "category:past_golden_globe_winners": "AWARD",
    "category:past_oscar_winners": "AWARD",
    "category:patricia_walden": "PERSON",
    "category:paul_mccartney": "PERSON",
    "category:pbs": "NETWORK_CHANNEL",
    "category:peaceful": "WEAK_DESCRIPTOR",
    "category:performing_arts": "STRONG_SEMANTIC",
    "category:philosophical": "WEAK_DESCRIPTOR",
    "category:pierce_brosnan": "PERSON",
    "category:pink_floyd": "PERSON",
    "category:pirates_of_the_caribbean": "FRANCHISE",
    "category:pixar": "PUBLISHER",
    "category:planet_of_the_apes": "FRANCHISE",
    "category:playful": "WEAK_DESCRIPTOR",
    "category:playing_shakespeare": "EDITION",
    "category:politics": "STRONG_SEMANTIC",
    "category:powerful": "WEAK_DESCRIPTOR",
    "category:presley,_elvis": "PERSON",
    "category:prime_video": "PLATFORM",
    "category:primetime_emmy®_central": "AWARD",
    "category:prince": "PERSON",
    "category:princess_diaries": "FRANCHISE",
    "category:prison": "STRONG_SEMANTIC",
    "category:queen": "PERSON",
    "category:queen_esther": "STRONG_SEMANTIC",
    "category:quirky": "WEAK_DESCRIPTOR",
    "category:ralph_bakshi": "PERSON",
    "category:reality_tv": "STRONG_SEMANTIC",
    "category:religion_&_spirituality": "STRONG_SEMANTIC",
    "category:rice,_damien": "PERSON",
    "category:rich,_buddy": "PERSON",
    "category:rieu,_andre": "PERSON",
    "category:ringo_starr": "PERSON",
    "category:riverdance": "FRANCHISE",
    "category:robots_&_androids": "STRONG_SEMANTIC",
    "category:rock,_chris": "PERSON",
    "category:rodney_yee": "PERSON",
    "category:roger_moore": "PERSON",
    "category:romance": "STRONG_SEMANTIC",
    "category:romantic": "WEAK_DESCRIPTOR",
    "category:roxy_music": "PERSON",
    "category:roy_orbison": "PERSON",
    "category:rush": "PERSON",
    "category:russia": "GEO_LANG",
    "category:sad": "WEAK_DESCRIPTOR",
    "category:santana": "PERSON",
    "category:sassy": "WEAK_DESCRIPTOR",
    "category:scary": "WEAK_DESCRIPTOR",
    "category:sci-fi_&_fantasy": "STRONG_SEMANTIC",
    "category:sci-fi_action": "STRONG_SEMANTIC",
    "category:sci-fi_series_&_sequels": "STRONG_SEMANTIC",
    "category:sci_fi_channel": "NETWORK_CHANNEL",
    "category:science_fiction": "STRONG_SEMANTIC",
    "category:science_fiction_&_fantasy": "STRONG_SEMANTIC",
    "category:scooby_doo": "FRANCHISE",
    "category:scooby_doo_animated_movies": "FRANCHISE",
    "category:scooby_doo_live_action_movies": "FRANCHISE",
    "category:sean_connery": "PERSON",
    "category:sensual": "WEAK_DESCRIPTOR",
    "category:sentimental": "WEAK_DESCRIPTOR",
    "category:serious": "WEAK_DESCRIPTOR",
    "category:shakespeare_101": "EDITION",
    "category:shakespeare_for_kids": "EDITION",
    "category:shakespeare_on_dvd_store": "EDITION",
    "category:shakur,_tupac": "PERSON",
    "category:showtime": "NETWORK_CHANNEL",
    "category:shrek": "FRANCHISE",
    "category:silent_films": "STRONG_SEMANTIC",
    "category:slasher": "STRONG_SEMANTIC",
    "category:snoop_dogg": "PERSON",
    "category:sober": "WEAK_DESCRIPTOR",
    "category:software": "NAVIGATION",
    "category:sony_pictures_classics": "PUBLISHER",
    "category:sony_pictures_home_entertainment": "PUBLISHER",
    "category:sophisticated": "WEAK_DESCRIPTOR",
    "category:space_adventure": "STRONG_SEMANTIC",
    "category:spain": "GEO_LANG",
    "category:spanish": "GEO_LANG",
    "category:spanish_language": "GEO_LANG",
    "category:spears,_britney": "PERSON",
    "category:special_editions": "FORMAT_META",
    "category:special_interest": "STRONG_SEMANTIC",
    "category:special_interests": "STRONG_SEMANTIC",
    "category:sports": "STRONG_SEMANTIC",
    "category:sports_&_outdoors": "NAVIGATION",
    "category:spotlight_deals": "PROMO_DEAL",
    "category:springsteen,_bruce": "PERSON",
    "category:stand_up": "STRONG_SEMANTIC",
    "category:star_wars": "FRANCHISE",
    "category:stephen_lynch": "PERSON",
    "category:sting": "PERSON",
    "category:stop-motion_&_clay_animation": "STRONG_SEMANTIC",
    "category:strange": "WEAK_DESCRIPTOR",
    "category:streisand,_barbra": "PERSON",
    "category:studio_specials": "PROMO_DEAL",
    "category:stunning": "WEAK_DESCRIPTOR",
    "category:subversive_cinema": "WEAK_DESCRIPTOR",
    "category:sundance_channel_home_entertainment": "PUBLISHER",
    "category:superman_animated_movies": "FRANCHISE",
    "category:suspense": "STRONG_SEMANTIC",
    "category:suzanne_deason": "PERSON",
    "category:sweden": "GEO_LANG",
    "category:swedish": "GEO_LANG",
    "category:sweet": "WEAK_DESCRIPTOR",
    "category:tai_chi": "STRONG_SEMANTIC",
    "category:taiwan": "GEO_LANG",
    "category:talk_show_and_variety": "STRONG_SEMANTIC",
    "category:talking_heads": "PERSON",
    "category:television": "UMBRELLA",
    "category:tenacious_d": "PERSON",
    "category:tense": "WEAK_DESCRIPTOR",
    "category:terminator": "FRANCHISE",
    "category:terrifying": "WEAK_DESCRIPTOR",
    "category:the_1950s": "TEMPORAL",
    "category:the_1960s": "TEMPORAL",
    "category:the_1970s": "TEMPORAL",
    "category:the_1980s": "TEMPORAL",
    "category:the_1990s": "TEMPORAL",
    "category:the_beatles": "PERSON",
    "category:the_big_dvd_&_blu-ray_blowout": "PROMO_DEAL",
    "category:the_big_dvd_sale": "PROMO_DEAL",
    "category:the_blues_brothers": "FRANCHISE",
    "category:the_comedies": "STRONG_SEMANTIC",
    "category:the_comedy_central_store": "NETWORK_CHANNEL",
    "category:the_doors": "PERSON",
    "category:the_histories": "STRONG_SEMANTIC",
    "category:the_rolling_stones": "PERSON",
    "category:the_shins": "PERSON",
    "category:the_temptations": "PERSON",
    "category:the_tragedies": "STRONG_SEMANTIC",
    "category:the_twilight_zone": "FRANCHISE",
    "category:the_who": "PERSON",
    "category:the_works": "COLLECTION",     # broad publisher-catalog dump → NoiseMeta
    "category:thoughtful": "WEAK_DESCRIPTOR",
    "category:thrilling": "WEAK_DESCRIPTOR",
    "category:timeless_holiday_favorites": "TEMPORAL",
    "category:timothy_dalton_&_george_lazenby": "PERSON",
    "category:tlc": "NETWORK_CHANNEL",
    "category:today's_deals": "PROMO_DEAL",
    "category:today's_spotlight_deals": "PROMO_DEAL",
    "category:tools_&_home_improvement": "NAVIGATION",
    "category:top_sellers": "COLLECTION",   # storefront popularity tag → NoiseMeta
    "category:touching": "WEAK_DESCRIPTOR",
    "category:toys_&_games": "NAVIGATION",
    "category:travel_channel": "NETWORK_CHANNEL",
    "category:turkey": "GEO_LANG",
    "category:turner,_tina": "PERSON",
    "category:tv": "UMBRELLA",
    "category:tv_&_miniseries": "STRONG_SEMANTIC",
    "category:tv_game_shows": "STRONG_SEMANTIC",
    "category:tv_news_programming": "STRONG_SEMANTIC",
    "category:tv_series": "STRONG_SEMANTIC",
    "category:tv_talk_shows": "STRONG_SEMANTIC",
    "category:twain,_shania": "PERSON",
    "category:tweens": "AGE_DEMO",
    "category:twilight_zone_dvds": "FRANCHISE",
    "category:two-disc_special_editions": "FORMAT_META",
    "category:ultimate_collector's_editions": "FORMAT_META",
    "category:ultimate_editions": "FORMAT_META",
    "category:understated": "WEAK_DESCRIPTOR",
    "category:united_kingdom": "GEO_LANG",
    "category:universal_studios_featured_deals": "PROMO_DEAL",
    "category:universal_studios_home_entertainment": "PUBLISHER",
    "category:unscripted": "STRONG_SEMANTIC",
    "category:unsettling": "WEAK_DESCRIPTOR",
    "category:upbeat": "WEAK_DESCRIPTOR",
    "category:upclose": "WEAK_DESCRIPTOR",
    "category:van_halen": "PERSON",
    "category:vaughan,_stevie_ray": "PERSON",
    "category:video_games": "NAVIGATION",
    "category:vietnam": "GEO_LANG",
    "category:visceral": "WEAK_DESCRIPTOR",
    "category:vista_series": "EDITION",
    "category:wall-e": "FRANCHISE",
    "category:walt_disney_legacy_collection": "EDITION",
    "category:walt_disney_studios_home_entertainment": "PUBLISHER",
    "category:walt_disney_treasures": "EDITION",
    "category:warner_archive": "EDITION",
    "category:warner_bros._featured_deals": "PROMO_DEAL",
    "category:warner_dvd_&_blu-ray_deals": "PROMO_DEAL",
    "category:warner_home_video": "PUBLISHER",
    "category:warner_sale": "PROMO_DEAL",
    "category:warner_video_bargains": "PROMO_DEAL",
    "category:waters,_muddy": "PERSON",
    "category:weird_al": "PERSON",
    "category:western": "STRONG_SEMANTIC",
    "category:westerns": "STRONG_SEMANTIC",
    "category:whimsical": "WEAK_DESCRIPTOR",
    "category:widescreen": "FORMAT_META",
    "category:wondrous": "WEAK_DESCRIPTOR",
    "category:x-men": "FRANCHISE",
    "category:yes": "PERSON",
    "category:yoga": "STRONG_SEMANTIC",
    "category:yoga_studios": "STRONG_SEMANTIC",
    "category:yoga_zone": "PUBLISHER",
    "category:young,_neil": "PERSON",
    "category:young_adult_audience": "AGE_DEMO",
}

# Roles that are valid as goal_concepts for short-term intent
GOAL_ELIGIBLE_ROLES: frozenset[str] = frozenset({
    "STRONG_SEMANTIC",
    "WEAK_DESCRIPTOR",  # allowed but lower priority than STRONG_SEMANTIC
    "AGE_DEMO",         # kids_&_family is a valid content direction
    "GEO_LANG",         # foreign_films, japanese cinema are valid directions
    "TEMPORAL",         # christmas movies etc. are a valid short-term direction
    "AWARD",            # oscar-winner browsing is a real intent
    "NETWORK_CHANNEL",  # user might be exploring bbc or hbo content specifically
    "FRANCHISE",        # task_focus: user actively seeking franchise content (star_wars, harry_potter, ...)
    "EDITION",          # task_focus: user seeking specific curated edition (criterion, box_sets, ...)
})

# Roles that should never appear as goal_concepts
GOAL_EXCLUDED_ROLES: frozenset[str] = frozenset({
    "PLATFORM",
    "NAVIGATION",
    "PROMO_DEAL",
    "PUBLISHER",
    "FORMAT_META",
    "COLLECTION",  # publisher catalog dumps and editorial noise → NoiseMeta
    "UMBRELLA",
    "PERSON",
})

# Strong-semantic roles: primary goal candidates for exploration
STRONG_GOAL_ROLES: frozenset[str] = frozenset({
    "STRONG_SEMANTIC",
    "AGE_DEMO",
    "GEO_LANG",
    "TEMPORAL",
    "AWARD",
    "NETWORK_CHANNEL",
})


def get_role(concept_id: str) -> str:
    """Return the role for a concept_id. Unknown concepts default to STRONG_SEMANTIC."""
    return CONCEPT_ROLES.get(concept_id, "STRONG_SEMANTIC")


def is_goal_eligible(concept_id: str) -> bool:
    """
    Return True if this concept is eligible as a goal_concept for short-term intent.
    Includes both STRONG_SEMANTIC and WEAK_DESCRIPTOR (plus AGE_DEMO, GEO_LANG, etc.).
    Concepts not in CONCEPT_ROLES (unknown) are treated as STRONG_SEMANTIC (eligible).
    Only applies to category: concepts -- format/price_band handled elsewhere.
    """
    if not concept_id.startswith("category:"):
        return True  # format/price_band managed by separate rules
    role = get_role(concept_id)
    return role in GOAL_ELIGIBLE_ROLES


def is_strong_semantic(concept_id: str) -> bool:
    """
    Return True if this concept is a strong-semantic (primary genre/subgenre) concept.
    WEAK_DESCRIPTOR concepts return False -- they are auxiliary, not primary goal direction.
    """
    if not concept_id.startswith("category:"):
        return False  # format/price_band are not semantic
    role = get_role(concept_id)
    return role in STRONG_GOAL_ROLES


# ── Ontology zone helpers (thin bridge to PGIMOntology) ──────────────────────
# These are lightweight wrappers that do NOT require loading the full ontology.
# Use for fast inline zone checks without instantiating PGIMOntology.

# Role → Zone mapping (mirrors _ROLE_TO_ZONE_SUBZONE in pgim_ontology.py)
# COLLECTION is omitted — requires per-concept disambiguation in PGIMOntology.
_ROLE_TO_ZONE: dict[str, str] = {
    "STRONG_SEMANTIC": "SemanticCore",
    "WEAK_DESCRIPTOR": "SemanticCore",
    "PLATFORM":        "ProductContext",
    "NAVIGATION":      "NoiseMeta",
    "PROMO_DEAL":      "NoiseMeta",
    "PUBLISHER":       "ProductContext",
    "FORMAT_META":     "ProductContext",
    "FRANCHISE":       "SemanticAnchor",   # named fictional universe (star_wars, harry_potter, ...)
    "EDITION":         "ProductContext",   # curated editorial/physical edition (criterion, box_sets, ...)
    "COLLECTION":      "NoiseMeta",        # publisher catalog dumps and storefront editorial noise
    "UMBRELLA":        "NoiseMeta",
    "PERSON":          "SemanticAnchor",
    "TEMPORAL":        "SemanticAnchor",
    "GEO_LANG":        "SemanticAnchor",
    "AGE_DEMO":        "SemanticCore",
    "AWARD":           "SemanticAnchor",
    "NETWORK_CHANNEL": "SemanticAnchor",
}


def get_ontology_zone(concept_id: str) -> str:
    """Fast zone lookup using role→zone mapping (no YAML load required).

    Returns one of: "SemanticCore" | "SemanticAnchor" | "ProductContext" |
                    "NoiseMeta" | "Unknown"

    FRANCHISE → SemanticAnchor, EDITION → ProductContext, COLLECTION → NoiseMeta.
    """
    role = get_role(concept_id)
    return _ROLE_TO_ZONE.get(role, "Unknown")


def is_semantic_core(concept_id: str) -> bool:
    """True if concept maps to SemanticCore zone."""
    return get_ontology_zone(concept_id) == "SemanticCore"


def is_noise_meta(concept_id: str) -> bool:
    """True if concept maps to NoiseMeta zone."""
    return get_ontology_zone(concept_id) == "NoiseMeta"


# ── Semantic Goal Hygiene ──────────────────────────────────────────────────────
# Hard-block non-semantic concepts from appearing in goal_concepts /
# validated_goal_concepts slots.  This layer sits BELOW is_goal_eligible():
# is_goal_eligible handles category: role hygiene; is_semantic_goal additionally
# blocks format:* and price_band:* prefixes which bypass the role taxonomy.
#
# Design:
#   - Prefix-level block:  format:*, price_band:*  always excluded from goal slot
#   - Role-level block:    PLATFORM / UMBRELLA / NAVIGATION / PROMO_DEAL /
#                          PUBLISHER / FORMAT_META / COLLECTION  (already in GOAL_EXCLUDED_ROLES)
#   - Unknown category concepts default to STRONG_SEMANTIC (eligible) per existing policy
#
# Usage:
#   is_semantic_goal(concept_id) -> bool
#   filter_non_semantic_goals(goals) -> (kept, removed, reasons)

_NON_SEMANTIC_PREFIXES: frozenset[str] = frozenset({
    "price_band",
    "format",
})


def is_semantic_goal(concept_id: str) -> bool:
    """
    Return True if concept_id is a valid semantic goal (allowed in goal_concepts slot).

    Hard-blocks:
      - price_band:* and format:* prefixes (technical/meta, not semantic intent)
      - category: concepts with GOAL_EXCLUDED_ROLES (PLATFORM, UMBRELLA, NAV, etc.)

    All other category: concepts (including unknown → STRONG_SEMANTIC) are allowed.
    Non-category, non-price_band, non-format concepts are allowed (conservative).
    """
    prefix = concept_id.split(":")[0]
    if prefix in _NON_SEMANTIC_PREFIXES:
        return False
    if prefix == "category":
        return is_goal_eligible(concept_id)
    return True  # other unknown prefixes: conservative allow


def filter_non_semantic_goals(
    goals: list[str],
    extra_blocklist: "list[str] | None" = None,
) -> "tuple[list[str], list[str], dict[str, str]]":
    """
    Filter a list of goal concept_ids, removing non-semantic concepts.

    Args:
        goals:          raw or validated goal concept_ids
        extra_blocklist: optional additional concept_ids to always remove

    Returns:
        kept    — semantic goals that passed the filter
        removed — concepts that were removed
        reasons — {concept_id: reason_string} for diagnostics
    """
    blocklist: set[str] = set(extra_blocklist) if extra_blocklist else set()
    kept: list[str] = []
    removed: list[str] = []
    reasons: dict[str, str] = {}

    for cid in goals:
        if cid in blocklist:
            removed.append(cid)
            reasons[cid] = "extra_blocklist"
            continue
        prefix = cid.split(":")[0]
        if prefix in _NON_SEMANTIC_PREFIXES:
            removed.append(cid)
            reasons[cid] = f"non_semantic_prefix:{prefix}"
            continue
        if prefix == "category":
            role = get_role(cid)
            if role in GOAL_EXCLUDED_ROLES:
                removed.append(cid)
                reasons[cid] = f"excluded_role:{role}"
                continue
        kept.append(cid)

    return kept, removed, reasons
