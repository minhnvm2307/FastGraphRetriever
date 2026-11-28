"""QA over retrieved triplets with the P2h prompt, using the DeepSeek chat API.

Everything (P2h prompt construction, API call, repair turn, answer parsing, timing, token counts) is in one function,
p2h_inference(). The prompt follows reason/qa.py --prompt P2h, with one repair turn for a broken response.

The API key is read from DEEPSEEK_API_KEY (or passed as api_key=).

  from demo.p2h_inference import p2h_inference
  out = p2h_inference(
      triplets=[('Lou Seal', 'sports.mascot.team', 'San Francisco Giants'), ...],   # top-K, best first
      question='What year did the team with mascot named Lou Seal win the World Series?',
      topic_entities=['Lou Seal'])
  out['answers'], out['response'], out['timing'], out['usage']
"""
# Local API credentials are loaded from the environment.

def p2h_inference(triplets, question, topic_entities, model='deepseek-flash',
                        api_key=None, max_tokens=4000, k=100, temperature=0.0, seed=0, repair=True,
                        base_url='https://api.deepseek.com', timeout=600, retries=5,
                        history=None):
    """Answer `question` from `triplets` with the P2h prompt on DeepSeek.

    triplets        list of (head, relation, tail), best first; extra fields (e.g. a score) are ignored; the first k are used
    question        the question text
    topic_entities  the question's topic entities (paths start there); P2h needs at least one that occurs in the triplets
    model           DeepSeek model id
    max_tokens      4000 for WebQSP, 1024 for CWQ in the paper

    Returns a dict:
      answers     the "ans:" answers, de-duplicated
      response    full reader text (the repair answer appended after a blank line if a repair turn was made)
      repaired    whether a repair turn was made, and why ('no_ans', 'too_long', 'length' or None)
      messages    the exact messages of the first call (system, in-context example, question)
      calls       per API call: role ('main' / 'repair'), finish_reason, latency_s, usage, request id, model
      usage       prompt / completion / total tokens summed over calls
      timing      build_prompt_s, api_s (sum of call latencies), total_s, completion tokens per second
      context     n_triplets used, n_paths shown, n_other_facts, n_candidates
    """
    import collections, os, re, time
    import requests

    t_start = time.perf_counter()
    key = api_key or os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise RuntimeError('set DEEPSEEK_API_KEY (or pass api_key=)')

    # ------------------------------------------------------------------ prompts (SubgraphRAG sys_icl_dc frame, as qa.py)
    SYS = ('Based on the triplets retrieved from a knowledge graph, please answer the question. '
           'Please return formatted answers as a list, each prefixed with "ans:".')
    P2_SYS = (SYS + ' The evidence is given as reasoning paths that start at the entity of the question.'
              ' First decide which chain of relations the question asks for, then answer with the entities at the end of the'
              ' paths that follow that chain and satisfy every condition of the question (time, type, role, location).'
              ' Several answers are allowed.')
    ICL_TRIPLES = [
        ('Lou Seal', 'sports.mascot.team', 'San Francisco Giants'),
        ('San Francisco Giants', 'sports.sports_team.championships', '2012 World Series'),
        ('San Francisco Giants', 'sports.sports_championship_event.champion', '2014 World Series'),
        ('San Francisco Giants', 'time.participant.event', '2014 Major League Baseball season'),
        ('San Francisco Giants', 'time.participant.event', '2010 World Series'),
        ('San Francisco Giants', 'time.participant.event', '2010 Major League Baseball season'),
        ('San Francisco Giants', 'sports.sports_team.championships', '2014 World Series'),
        ('San Francisco Giants', 'sports.sports_team.team_mascot', 'Crazy Crab'),
        ('San Francisco Giants', 'sports.sports_team.championships', '2010 World Series'),
        ('San Francisco Giants', 'sports.professional_sports_team.owner_s', 'Bill Neukom'),
        ('San Francisco Giants', 'time.participant.event', '2012 World Series'),
        ('San Francisco', 'sports.sports_team_location.teams', 'San Francisco Giants'),
        ('San Francisco Giants', 'sports.sports_team.arena_stadium', 'AT&T Park'),
        ('AT&T Park', 'location.location.events', '2012 World Series'),
        ('m.011zsc4_', 'organization.leadership.organization', 'San Francisco Giants'),
        ('San Francisco Giants', 'sports.sports_team.previously_known_as', 'New York Giants'),
        ('AT&T Park', 'location.location.events', '2010 World Series'),
        ('Crazy Crab', 'sports.mascot.team', 'San Francisco Giants'),
        ('New York Giants', 'baseball.baseball_team.league', 'National League'),
        ('San Francisco Giants', 'sports.sports_team.colors', 'Black'),
        ('San Francisco Giants', 'sports.sports_team.previously_known_as', 'New York Gothams'),
        ('m.0k079qm', 'base.schemastaging.team_training_ground_relationship.team', 'San Francisco Giants'),
        ('m.0k079ry', 'base.schemastaging.team_training_ground_relationship.team', 'San Francisco Giants'),
        ('2010 World Series', 'time.event.locations', 'AT&T Park'),
        ('San Francisco Giants', 'time.participant.event', '2012 Major League Baseball season'),
        ('San Francisco Giants', 'baseball.baseball_team.league', 'National League'),
        ('m.0crtd80', 'sports.sports_league_participation.league', 'National League West'),
        ('San Francisco Giants', 'sports.sports_team.location', 'San Francisco'),
        ('San Francisco Giants', 'sports.sports_team.sport', 'Baseball'),
        ('m.05n6dtn', 'baseball.baseball_team_stats.team', 'San Francisco Giants')]
    ICL_QUESTION = 'What year did the team with mascot named Lou Seal win the World Series?'
    ICL_ANSWER = (
        'To find the year the team with mascot named Lou Seal won the World Series, we need to find the team with mascot '
        'named Lou Seal and then find the year they won the World Series.\n\n'
        'From the paths, we can see that Lou Seal is the mascot of the San Francisco Giants.\n\n'
        'Now, we need to find the year the San Francisco Giants won the World Series.\n\n'
        'From the paths, we can see that San Francisco Giants won the 2010 World Series and 2012 World Series and '
        '2014 World Series.\n\n'
        'So, the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012, and 2014.\n\n'
        'Therefore, the formatted answers are:\n\n'
        'ans: 2014 (2014 World Series)\nans: 2012 (2012 World Series)\nans: 2010 (2010 World Series)')
    NO_ANS_FIX = ('Your previous answer did not end with any line prefixed "ans:". Based on the evidence and your reasoning '
                  'above, give only the final answers, one per line, each prefixed with "ans:".')
    LOOP_FIX = ('Your previous answer was cut off because it started repeating itself. Based on the evidence and your '
                'reasoning above, give only the final answers, one per line, each prefixed with "ans:".')
    CVT = re.compile(r'^[mg]\.[0-9a-z_]+$')                     # Freebase mediator (CVT) node ids
    MAX_HOPS, MAX_PATHS, MAX_CANDIDATES, MAX_CLEAN_CHARS = 3, 40, 30, 6000

    def short(rel):
        return rel.split('.')[-1].replace('_', ' ')

    def walk(T, topics, full_key=False):
        """simple paths (<= 3 hops, both directions) from a topic; CVT nodes are passed through but never end a path;
        ranked by their worst triple, then by length: [(worst, nodes, [(relation, reverse)])]. A path is kept once per
        displayed form (full_key=False) or once per full relation sequence (full_key=True), as qa.paths / qa.path_triples"""
        adj = collections.defaultdict(list)
        for i, (h, r, t) in enumerate(T):
            adj[h].append((t, r, False, i)); adj[t].append((h, r, True, i))
        out, seen = [], set()
        for s in topics:
            if s not in adj:
                continue
            stack = [(s, [s], [], 0)]
            while stack:
                node, nodes, steps, worst = stack.pop()
                if steps and not CVT.match(node):
                    key = (tuple(n for n in nodes if not CVT.match(n)),
                           tuple(steps) if full_key else tuple((short(r), rev) for r, rev in steps))
                    if key not in seen:
                        seen.add(key); out.append((worst, nodes, steps))
                if len(steps) >= MAX_HOPS:
                    continue
                for v, r, rev, i in adj[node]:
                    if v not in nodes:
                        stack.append((v, nodes + [v], steps + [(r, rev)], max(worst, i)))
        out.sort(key=lambda x: (x[0], len(x[2])))
        return out

    def user_turn(T, topics, q, other_facts):
        lines, ends, used = [], [], set()
        for _, nodes, steps in walk(T, topics)[:MAX_PATHS]:                    # the displayed paths
            parts = [nodes[0]]
            for n, (r, rev) in zip(nodes[1:], steps):
                parts.append(f'-> {short(r)}{" (of)" if rev else ""} ->'); parts.append('[...]' if CVT.match(n) else n)
            lines.append(' '.join(parts).replace('-> [...] ->', '->'))
            if nodes[-1] not in ends:
                ends.append(nodes[-1])
        seen, n_shown = set(), 0                                              # the triples they use (as qa.p2h_messages)
        for _, nodes, steps in walk(T, topics, full_key=True):
            key = (tuple(n for n in nodes if not CVT.match(n)), tuple((short(r), rev) for r, rev in steps))
            if key in seen:
                continue
            seen.add(key); n_shown += 1
            for a, b, (r, rev) in zip(nodes, nodes[1:], steps):
                used.add((b, r, a) if rev else (a, r, b))
            if n_shown == MAX_PATHS:
                break
        text = ('Reasoning paths (from the question entity, most relevant first):\n'
                + '\n'.join(f'{i + 1}. {l}' for i, l in enumerate(lines))
                + '\n\nCandidate answers (path ends): ' + '; '.join(ends[:MAX_CANDIDATES]))
        rest = [t for t in T if t not in used] if other_facts else []
        if rest:
            text += '\n\nOther facts:\n' + '\n'.join(f'({h},{r},{t})' for h, r, t in rest)
        q = q if q.endswith('?') else q + '?'
        return text + '\n\n\nQuestion:\n' + q, len(lines), len(rest), min(len(ends), MAX_CANDIDATES)

    T = [tuple(str(x) for x in t[:3]) for t in triplets[:k]]
    topics = [str(e) for e in (topic_entities if isinstance(topic_entities, (list, tuple)) else [topic_entities])]
    icl_user, _, _, _ = user_turn(ICL_TRIPLES, ['Lou Seal'], ICL_QUESTION, other_facts=False)
    test_user, n_paths, n_other, n_cand = user_turn(T, topics, question, other_facts=True)
    prior_turns = [
        {'role': item['role'], 'content': item['content'][:2000]}
        for item in (history or [])[-6:]
        if item.get('role') in ('user', 'assistant') and isinstance(item.get('content'), str)
    ]
    messages = [{'role': 'system', 'content': P2_SYS},
                {'role': 'user', 'content': icl_user},
                {'role': 'assistant', 'content': ICL_ANSWER},
                *prior_turns,
                {'role': 'user', 'content': test_user}]
    t_built = time.perf_counter()

    # ------------------------------------------------------------------------------------------------ API calls
    url = base_url.rstrip('/') + '/chat/completions'
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    calls = []

    def call(msgs, n_tokens, role):
        body = {'model': model, 'messages': msgs, 'temperature': temperature,
                'thinking': {'type': 'disabled'}, 'max_tokens': n_tokens}
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                r = requests.post(url, json=body, headers=headers, timeout=timeout)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(2 * (attempt + 1)); continue
                r.raise_for_status()
                j = r.json()
                break
            except (requests.ConnectionError, requests.Timeout, ValueError):
                if attempt == retries:
                    raise
                time.sleep(2 * (attempt + 1))
        latency = time.perf_counter() - t0
        choice = j['choices'][0]
        calls.append({'role': role, 'finish_reason': choice.get('finish_reason'), 'latency_s': round(latency, 3),
                      'usage': j.get('usage', {}), 'id': j.get('id'), 'model': j.get('model', model),
                      'attempts': attempt + 1})
        return choice['message'].get('content') or '', choice.get('finish_reason')

    def cut_loop(text, min_len=20):
        seen = set()
        for m in re.finditer(r'[^\n.]+(?:[\n.]|$)', text):
            s = ' '.join(m.group().split()).lower()
            if len(s) >= min_len:
                if s in seen:
                    return text[:m.start()].rstrip()
                seen.add(s)
        return text

    response, finish = call(messages, max_tokens, 'main')
    why = ('length' if finish == 'length' else 'no_ans' if 'ans:' not in response
           else 'too_long' if len(response) > MAX_CLEAN_CHARS else None)
    if repair and why:
        shown = cut_loop(response) if finish == 'length' else response
        tail, _ = call(messages + [{'role': 'assistant', 'content': shown},
                                   {'role': 'user', 'content': LOOP_FIX if finish == 'length' else NO_ANS_FIX}],
                       256, 'repair')
        response = shown + '\n\n' + tail

    # --------------------------------------------------------------------------------------------- answers & stats
    lines = [p for p in response.split('\n') if 'ans:' in p and 'none' not in p.lower()]
    lines = [p for p in lines if 'ans: not available' not in p.lower() and 'ans: no information available' not in p.lower()]
    answers = []
    for p in lines:
        a = re.sub(r'^\s*ans:\s*', '', p).strip()
        if a and a not in answers:
            answers.append(a)

    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    cost = None
    for c in calls:
        for f in usage:
            usage[f] += int(c['usage'].get(f) or 0)
        if c['usage'].get('estimated_cost') is not None:
            cost = (cost or 0.0) + float(c['usage']['estimated_cost'])
    if cost is not None:
        usage['estimated_cost'] = round(cost, 8)
    api_s = sum(c['latency_s'] for c in calls)
    t_end = time.perf_counter()
    return {'question': question, 'answers': answers, 'response': response, 'repaired': why if (repair and why) else None,
            'messages': messages, 'calls': calls, 'usage': usage,
            'timing': {'build_prompt_s': round(t_built - t_start, 4), 'api_s': round(api_s, 3),
                       'total_s': round(t_end - t_start, 3),
                       'completion_tokens_per_s': round(usage['completion_tokens'] / api_s, 1) if api_s else None},
            'context': {'n_triplets': len(T), 'n_paths': n_paths, 'n_other_facts': n_other, 'n_candidates': n_cand},
            'model': model}


if __name__ == '__main__':
    import json, sys
    demo = p2h_inference(
        triplets=[('Lou Seal', 'sports.mascot.team', 'San Francisco Giants'),
                  ('San Francisco Giants', 'sports.sports_team.championships', '2014 World Series'),
                  ('San Francisco Giants', 'sports.sports_team.championships', '2012 World Series'),
                  ('San Francisco Giants', 'sports.sports_team.location', 'San Francisco')],
        question='Which championships did the team with the mascot Lou Seal win',
        topic_entities=['Lou Seal'], max_tokens=int(sys.argv[1]) if len(sys.argv) > 1 else 1024)
    demo.pop('messages')
    print(json.dumps(demo, indent=2, ensure_ascii=False))
