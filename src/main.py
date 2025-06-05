import pandas as pd

from utilities.basicLLama import BasicLLama
from utilities.utils import parse_json, calculate_age
from utilities.promptCreation import PromptCreation, PersonaCall

from typing import Dict

import random
import argparse

import ast

import os

import tqdm

VOTE_OPTIONS: list[str] = ["FOR", "AGAINST", "ABSTENTION"]

JSON_COT_ATTRIBUTES: list[str] = ["reasoning", "vote"]
JSON_COT_EXPLANATION: list[str] = ["Why you choose to vote this way.", "Based on your reasoning cast your vote. Response can only be from these values: " + "|".join(VOTE_OPTIONS)]

JSON_PRO_CONTRA_ATTRIBUTES: list[str] = ["pro", "contra", "evaluation", "vote"]
JSON_PRO_CONTRA_EXPLANATION: list[str] = ["Provide your reasoning to vote in favor of the proposal.", "Provide your reasoning to vote against the proposal.", "Evaluate both pro and contra reasons and reason if one side is more important to you." , "Based on your reasoning cast your vote. Response can only be from these values: " + "|".join(VOTE_OPTIONS)]

JSON_NO_REASONING_ATTRIBUTES: list[str] = ["vote"]
JSON_NO_REASONING_EXPLANATION: list[str] = ["Cast your vote. Response can only be from these values: " + "|".join(VOTE_OPTIONS)]

COUNTRY: str = "country"
GROUP: str = "group"
AGE: str = "age"
GENDER: str = "gender"
BIRTHPLACE: str = "birthplace"
NATIONAL_PARTY: str = "national_party"
ALL_ATTRIBUTES: list[str] = [GENDER, AGE, BIRTHPLACE, GROUP, NATIONAL_PARTY, COUNTRY]

MEP_PERSONA: str = "A member of the european parliament."
DEFAULT_POLITICIAN: str = "A politician."
US_POLITICIAN: str = "An american politician."
DEFAULT_PERSON: str = "A person from europe."

DEFAULT_PERSONAS: list[str] = [MEP_PERSONA, DEFAULT_POLITICIAN, US_POLITICIAN, DEFAULT_PERSON]

class EuropeanVoter:

    SYSTEM_PROMPT: str = "Your task is to act exactly like a politician you will be given."

    def __init__(self, vote_id: int, vote_title: str, give_speeches:bool, json_attributes:list[str], json_explanation:list[str], give_wiki:bool = True, roll_call_vote:bool = True, attribute_list: list[str] = [], default_personas: bool = False, counterfactual_speeches: bool = False, no_persona: bool= False, opposition: bool = False, modified_speeches: bool = True):
        self._vote_id: int = vote_id
        self._give_wiki: bool = give_wiki
        self._default_personas: bool = default_personas
        self._counterfactual_speeches: bool = counterfactual_speeches
        self._opposition: bool = opposition

        #ADD ARGUMENT SUPPORT
        self._modified_speeches = modified_speeches

        self._no_persona : bool = no_persona
        if default_personas:
            self._give_wiki = False
        self._speeches: list[str] = self._find_relevant_speeches()
        self._vote_title: str = vote_title
        self._voters: pd.DataFrame = self._find_relevant_voters()
        self._give_speeches: bool = give_speeches
        self._json_attributes: list[str] = json_attributes
        self._json_explanation: list[str] = json_explanation
        self._roll_call_vote: bool = roll_call_vote
        self._attribute_list: list[str] = attribute_list
        self._generate_prompts_for_vote()

    def _find_relevant_speeches(self) -> list[str]:
        #df_speeches = pd.read_csv("../own_data/debate_xmls/all_debates_after_brexit_translated.csv")
        # df_speeches = pd.read_csv("../own_data/debate_xmls/all_debates_after_brexit_translated_ids.csv")
        # df_speeches = df_speeches[df_speeches["speaker_type"] == "au nom du groupe"]
        # df_speeches['vote_id'] = df_speeches['vote_id'].apply(lambda x: ast.literal_eval(x))
        # df_speeches = df_speeches[df_speeches['vote_id'].apply(lambda x: self._vote_id in x)]
        # speeches = df_speeches["translated_speech"].to_list()
        if self._modified_speeches:
            df_speeches = pd.read_csv("speeches_for_vote_modified.csv")
        else:
            df_speeches = pd.read_csv("counterfactual_speeches_with_title.csv")
        df_speeches = df_speeches[df_speeches['id'] == self._vote_id]
        
        if self._modified_speeches:
            speeches = df_speeches["modified_speech"].to_list()
        elif self._counterfactual_speeches:
            speeches = df_speeches["counter_speech"].to_list()
        else:
            speeches = df_speeches["translated_speech"].to_list()

        #print(speeches)
        if len(speeches) == 0:
            raise RuntimeError("No speeches found")
        random.shuffle(speeches)
        return speeches

    def _find_relevant_voters(self) -> pd.DataFrame:
        if self._give_wiki:
            df_prompt = pd.read_csv("../own_data/members/prompts_2019.csv")
            df_member_votes = pd.read_csv("../howTheyVoteDataSet/member_votes.csv")
            df_specific_votes: pd.DataFrame = df_member_votes[df_member_votes["vote_id"] == self._vote_id]
            df_result = df_specific_votes.merge(df_prompt, left_on="member_id", right_on="id")
            df_result = df_result[df_result["position"] != "DID_NOT_VOTE"]
        elif self._default_personas:
            persona_list = []
            for persona in DEFAULT_PERSONAS:
                persona_list.append([persona, self._vote_id])
            
            df_result = pd.DataFrame(persona_list, columns=["persona", "vote_id"])
        elif self._no_persona:
            persona_list = []
            persona_list.append(["no persona", self._vote_id])
            
            df_result = pd.DataFrame(persona_list, columns=["persona", "vote_id"])
        else:
            df_members = pd.read_csv("../own_data/members/members_voting_time.csv")
            df_member_votes = pd.read_csv("../howTheyVoteDataSet/member_votes.csv")
            df_votes = pd.read_csv("../howTheyVoteDataSet/votes.csv")
            row_votes = df_votes[df_votes["id"] == self._vote_id].iloc[0]
            timestamp = pd.to_datetime(row_votes["timestamp"])
            timestamp = timestamp.date()
            df_members["start_date"] = pd.to_datetime(df_members["start_date"]).dt.date
            df_members["end_date"] = pd.to_datetime(df_members["end_date"]).dt.date
            df_members = df_members[(df_members["start_date"] <= timestamp) & ((df_members["end_date"] >= timestamp))]
            df_specific_votes: pd.DataFrame = df_member_votes[df_member_votes["vote_id"] == self._vote_id]
            df_result = df_specific_votes.merge(df_members, left_on="member_id", right_on="member_id")
            #print(df_result.columns)
            df_result = df_result[df_result["position"] != "DID_NOT_VOTE"] 
        return df_result


    def _create_persona_description(self, row:pd.Series) -> str:
        df_votes = pd.read_csv("../howTheyVoteDataSet/votes.csv")
        row_votes = df_votes[df_votes["id"] == self._vote_id].iloc[0]

        # Convert to datetime
        timestamp = pd.to_datetime(row_votes["timestamp"])
        birthdate = pd.to_datetime(row["date_of_birth"])

        age = calculate_age(birthdate, timestamp)

        #There are only male and female MEPs in the ninth parliament
        gender_call = "He"
        if row["gender"] == "female":
            gender_call = "She"

        national_party = row["national_party"]

        national_party_sentence = ""
        if national_party == "Independent":
            national_party_sentence = f"{gender_call} is not part of a national party. "
        else:
            national_party_sentence = f"{gender_call} is part of the national party {national_party}. "

        national_party_sentence = national_party_sentence if NATIONAL_PARTY in self._attribute_list else ""

        country = row["country"]

        country_sentence = f"{gender_call} is representing the country {country}. "

        country_sentence = country_sentence if COUNTRY in self._attribute_list else ""

        group = row["group_name"]

        group_sentence = f"{gender_call} is part of "

        if group != "The Left group in the European Parliament - GUE/NGL":
            group_sentence += "the "

        group_sentence += f"{group}. "

        if group == "European Conservatives and Reformists Group" \
            or group == "The Left group in the European Parliament - GUE/NGL" \
                or group == "Identity and Democracy Group":
            opposition_sentence = "The group currently is in the opposition, so the vote of this politician might not influence the final outcome."
        elif group == "Non-attached Members":
            opposition_sentence = ""
        else:
            opposition_sentence = ""
        
        if self._opposition:
            group_sentence += f"{opposition_sentence}"

        group_sentence = group_sentence if GROUP in self._attribute_list else ""

        if age == pd.NaT:
            age_sentence = f", age unknown" if AGE in self._attribute_list else ""
        else:
            age_sentence = f", who is {age} years old" if AGE in self._attribute_list else ""    

        gender = row["gender"]

        gender_sentence = f", a {gender} politician" if GENDER in self._attribute_list else ""

        birthplace = row["birthplace"]

        birthplace_sentence = f", born in {birthplace}" if BIRTHPLACE in self._attribute_list else ""

        politician_description =  f"""{row["fullname"]}{gender_sentence}{age_sentence}{birthplace_sentence}. {country_sentence}{national_party_sentence}{group_sentence}"""

        politician_description = politician_description.strip()

        return politician_description

        # if just_name:
        #     politician_prompt = politician_prompt + POLITICIAN_NAME
        # else:
        #     politician_prompt = politician_prompt + row["prompt"]

        # if give_party:
        #     politician_prompt = politician_prompt + GROUP_CODE

    def _create_task_instruction(self) -> str:
        topic_description = """
You are voting on a proposition with the title: """ + self._vote_title
        
        if self._roll_call_vote:
            topic_description += """
The vote will be held as a roll call vote, so your vote will be visible to the public and to your party."""

        topic_summary = ""

        if self._give_speeches:
            topic_description = topic_description + """ 
Other politician have given the following speeches:"""
            topic_description = topic_description + "\n\n" + "\n\n".join(self._speeches)
        else:
            topic_description = topic_description + """ Key points of the proposition are:
""" + topic_summary
        
        return topic_description

    def _generate_prompts_for_vote(self) -> None:
        all_prompts = []
        for i, row in self._voters.iterrows():
            creator = PromptCreation()
            if self._no_persona:
                pass
            elif self._default_personas:
                creator.create_persona(description=row["persona"], persona_call=PersonaCall.ACT, persona_call_freetext= "the following person.")
            elif self._give_wiki:
                creator.create_persona(description=row["prompt"], persona_call=PersonaCall.ACT, persona_call_freetext= "the following politician.")
            else:
                creator.create_persona(description= self._create_persona_description(row), persona_call=PersonaCall.ACT, persona_call_freetext="the following politician.")
            
            creator.set_task_instruction(self._create_task_instruction())
            creator.set_ouput_format_json(json_attributes=self._json_attributes, json_explanation=self._json_explanation)
            all_prompts.append(creator.generate_prompt())
        print(all_prompts[0])
        self._voters["full_instruction"] = all_prompts

    def vote(self, llama:BasicLLama, repeat_per_person:int = 3, retries:int =3, batch_generation:bool = False, verbose:bool = False, temperature: float = 0.6, use_vllm: bool = False) -> pd.DataFrame:
        vote_result = []
        for i, row in tqdm.tqdm(self._voters.iterrows(), total=self._voters.shape[0], position=0, leave=True):
            parsed_dict = {attribute: [] for attribute in self._json_attributes}

            #reasonings = []
            #results = []
            
            if self._no_persona:
                self.SYSTEM_PROMPT = "You are a helpful assistant."
            if batch_generation:
                correct_responses:int = 0
                for _ in range(retries):

                    system_messages = [self.SYSTEM_PROMPT] * (repeat_per_person - correct_responses)
                    prompts = [row["full_instruction"]] * (repeat_per_person - correct_responses)
                    try:
                        raw_responses = llama.batch_generation(system_messages=system_messages, prompts=prompts, max_new_tokens=512, temperature = temperature)
                        for raw_resp in raw_responses:
                            #print(raw_resp)
                            if use_vllm:
                                raw_resp_content = raw_resp
                            else:
                                raw_resp_content = raw_resp["content"]
                            if verbose:
                                print(raw_resp_content, flush=True)
                            #print(raw_resp_content, flush=True)
                            parsed_json = parse_json(raw_resp_content)

                            if parsed_json == None:
                                raise Exception("Did not answer!")
                            for attribute in self._json_attributes:
                                parsed_dict[attribute].append(parsed_json[attribute])
                            correct_responses += 1
                        if correct_responses == repeat_per_person:
                            break
                    except Exception as e:
                        print(f"Retrying due to Exception: {e}")
            else:
                for _ in range(repeat_per_person):
                    for _ in range(retries):
                        raw_resp = None
                        try:
                            raw_resp = llama.basic_generation(system_message=self.SYSTEM_PROMPT, prompt=row["full_instruction"], max_new_tokens=512)["content"]
                            if verbose:
                                print(raw_resp, flush=True)

                            parsed_json = parse_json(raw_resp)

                            if parsed_json == None:
                                raise Exception("Did not answer!")

                            for attribute in self._json_attributes:
                                parsed_dict[attribute].append(parsed_json[attribute])
                            break

                        except Exception as e:
                            print(f"Retrying due to Exception: {e}")
            
            if self._default_personas:
                result_list = [self._vote_id, row["persona"], *parsed_dict.values()]
            elif self._no_persona:
                result_list = [self._vote_id, *parsed_dict.values()]
            elif not self._give_wiki:
                result_list = [row["member_id"], row["vote_id"], row["fullname"], *parsed_dict.values(), row["position"], row["group_code_x"], row["country_code"]]
            else:
                result_list = [row["id"], self._vote_id, row["fullName"], *parsed_dict.values(), row["position"], row["group_code"], row["country_code"]]

            #vote_result.append([row["id"], row["vote_id"], row["fullName"], reasonings, results, row["position"], row["group_code"], row["country_code"]])
            vote_result.append(result_list)

            named_list = [
            "reasonings" if item == "reasoning" else "llm_votes" if item == "vote" else item
            for item in self._json_attributes
            ]
        if self._no_persona:
            df_vote_results = pd.DataFrame(vote_result, columns=["vote_id", *named_list])
        elif self._default_personas:
            df_vote_results = pd.DataFrame(vote_result, columns=["vote_id", "persona", *named_list])
        else:    
            df_vote_results = pd.DataFrame(vote_result, columns=["member_id", "vote_id", "fullName", *named_list, "actual_vote", "group", "country"])
        #df_vote_results = pd.DataFrame(vote_result, columns=["member_id", "vote_id", "fullName", "reasonings", "llm_votes", "actual_vote", "group", "country"])
        return df_vote_results

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create LLM Vote Predictions.'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-3.3-70B-Instruct',
        help='The model that will be used for inference.'
    )

    parser.add_argument(
        '--output', 
        type=str,
        default="../own_data/votes/", 
        help='The output path.'
    )

    parser.add_argument(
        '--reasoning',
        type=str,
        default="reasoning", 
        help='The type of reasoning that should be used.'
    )

    parser.add_argument(
        '--wiki', 
        type=bool,
        default=False,
        help="If true, the wikipedia prompts will be used.",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--default_personas',
        default=False,
        help="If true, the default personas will be used instead.",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--roll_call', 
        type=bool,
        default=False,
        help="If true, the llm will be instructed that the vote is public.",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--counter_speeches', 
        type=bool,
        default=False,
        help="If true, the counterfactual speeches will be given to the LLM.",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--only_task', 
        type=bool,
        default=False,
        help="If true, no persona will be given as task.",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--attribute_list",
        nargs="*",
        default=ALL_ATTRIBUTES,
        type=str,
        help=f"Attributes that will be used in conjunction. Possible values are: {ALL_ATTRIBUTES}"
    )

    parser.add_argument(
        "--vote_list",
        nargs="*",
        default=[],
        type=str,
        help= "If a list is given, the vote will be restricted to these votes."
    )

    parser.add_argument(
        "--temperature",
        default=0.6,
        type=float,
        help= "Temperature_hyperparameter of the LLM"
    )

    parser.add_argument(
        '--opposition', 
        type=bool,
        default=False,
        help="If true, politicians of the opposition will be told to consider ABSTENTIONS.",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        '--modified_speeches', 
        type=bool,
        default=True,
        help="If true, the regex filtered speeches will be used.",
        action=argparse.BooleanOptionalAction
    )

    return parser.parse_args()

if __name__ == "__main__":
    random.seed(42)
    args = parse_arguments()

    if args.modified_speeches:
        print("running modified speeches")
    
    df_votes = pd.read_csv("../howTheyVoteDataSet/votes.csv")

    df_votes['timestamp'] = pd.to_datetime(df_votes['timestamp'], errors='coerce')

    cutoff_date = pd.Timestamp('2024-01-01')

    df_votes = df_votes[df_votes['timestamp'] >= cutoff_date]

    df_votes = df_votes[(df_votes["is_main"] == True) & (df_votes["is_featured"] == True)]

    json_attributes = JSON_COT_ATTRIBUTES
    json_explanation = JSON_COT_EXPLANATION

    if args.reasoning == "no_reasoning":
        json_attributes = JSON_NO_REASONING_ATTRIBUTES
        json_explanation = JSON_NO_REASONING_EXPLANATION
    if args.reasoning == "pro_contra":
        json_attributes = JSON_PRO_CONTRA_ATTRIBUTES
        json_explanation = JSON_PRO_CONTRA_EXPLANATION

    print(json_attributes)

    print(args.attribute_list)

    llama = BasicLLama(model_id=args.model, vllm_inference=args.vllm)

    print(args.vote_list)

    vote_id_ints = list(map(int, args.vote_list))

    df_speeches = pd.read_csv("counterfactual_speeches_with_title.csv")
    df_votes = df_votes[df_votes["id"].isin(df_speeches["id"])]

    if len(args.vote_list) > 0:
        df_votes = df_votes[df_votes["id"].isin(vote_id_ints)]
        #print(df_votes)
        
    for i, row in tqdm.tqdm(df_votes.iterrows(), total=df_votes.shape[0]):
        try:
            voter = EuropeanVoter(vote_id=int(row["id"]), vote_title=row["display_title"], give_speeches=True, json_attributes=json_attributes, json_explanation=json_explanation, give_wiki=args.wiki, roll_call_vote=args.roll_call, attribute_list=args.attribute_list, default_personas=args.default_personas, counterfactual_speeches=args.counter_speeches, no_persona=args.only_task, opposition=args.opposition, modified_speeches=args.modified_speeches)
        except Exception as e:
            print(str(row["id"]) + " not working because:" + str(e))
            continue
        path_reasoning = args.reasoning

        try:
            df_vote_results = voter.vote(llama, batch_generation=True, verbose=False, temperature = args.temperature)
        except Exception as e:
            print(str(row["id"]) + " vote not working because:" + str(e))
            continue
        
        path_description = ""
        if args.default_personas:
            path_description = "default_personas"
        elif args.only_task:
            path_description = "no_persona"
        elif args.wiki:
            path_description = "wikipedia_prompt"
        else:
            args.attribute_list.sort()
            ALL_ATTRIBUTES.sort()
            if args.attribute_list == ALL_ATTRIBUTES:
                path_description = "attribute_prompt_all"
            else:
                path_description = "attribute_prompt"
                for attribute in args.attribute_list:
                    path_description += f"_{attribute}"

        if args.roll_call:
            roll_call_path = "roll_call"
        else:
            roll_call_path = "no_roll_call"

        if args.counter_speeches:
            counter_speeches_path = "counter_speeches"
        elif args.modified_speeches:
            counter_speeches_path = "modified_speeches"
        else:
            counter_speeches_path = "real_speeches"

        temperature_path = str(args.temperature).replace(".", "_")

        if args.opposition:
            opposition_path = "opposition"
        else:
            opposition_path = "no_opposition"

        output_dir = args.output + "/" + args.model + "/" + path_reasoning + "/" + path_description + "/" + roll_call_path + "/" + counter_speeches_path + "/" + temperature_path + "/" + opposition_path
        output_file = "llm_votes_" + str(row["id"]) + ".csv"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df_vote_results.to_csv(output_dir + "/" + output_file, index=False)
