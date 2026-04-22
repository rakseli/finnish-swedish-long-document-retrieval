# Finnish-Swedish-Long-Document-Retrieval

1. Download HPLT v3 Finnish and Swedish score 10 and 9
2. Count tokens and sample long documents
3. Annotate the 1.5K dev test documents manually based on first 1000 chars
    - reject:
        - porn
        - advertisement such as betting
        - clearly machine translated
    - accept:
        - texts that seem otherwise good and consistent
    - for Swedish
        - the annotator has CEFR B1-level language profiency
        - use URLs for additional support for finding good texts
    - for Finnish
        - the annotator is native speaker
2. Split the good docs into paragraphs based on XML-tags and two consecutive newlines ("\n\n") and filter:
    - documents that coudn't be splitted into paragraphs using the tags
    - documents of which paragraphs splitting is too sparse --> 80% of token mass is in one paragraph
    - paragraphs that were less than 10 tokens
3. Sample 5 paragraphs per document and generate questions
    - for documents less than 5 paragraphs take all in random order
4. Annotate the questions using skimming until all docs have one good question
    - for Finnish:
        - reject:
            - questions that contain the answer
            - questions that are too generic e.g. "What is the heading of this text?", "Where was the event located?"
            - questions that contained "toxic" content e.g. racism or outdated vocabulary
            - question or text in wrong language
            - questions querying sensitive information
        - accept:
            - questions that seem reasonable with max 10s inspection
            - texts that seem reasonable with max 20s inspection
    - for Swedish
        - reject
            - questions that are too generic e.g. "What is the heading of this text?"
            - questions that contained "toxic" content e.g. racism
            - question or text in wrong language
            - questions querying sensitive information
        - accept
            - questions that seem reasonable with max 10s inspection
            - texts that seem reasonable with max 20s inspection
        - all questions were translated into English (using the same model as generation) to support annotation as the annotator is non-native
    - questions contain
        - need-in-haysack
        - NLU
        - different level of detail
    - better prompt would have been to mimic human search as some of the questions are too specific which do not reflect the information retrival meaningful for humans &rarr; we suggest to do this in future work
        - E.g. this doesn't seem something like a human would ask, rather a test question: "What ten specific measures for development within Finnish disability policy are listed in the Government's report on disability policy from 2006?"
