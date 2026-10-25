# Dobby Features

This document describes, in plain language, everything Dobby can do today. It is written from the point of view of the person using the robot, not the person programming it.

> **Maintenance:** whenever a feature is added, changed or removed, update this file as part of the same change.

## Voice conversation

- **Call it by name.** Just say "Dobby" (or "Reachy") and it starts listening. No button is needed, and the robot notices its name without an internet connection.
- **Natural conversation, in Brazilian Portuguese.** It answers in short sentences, and you can interrupt it mid-sentence.
- **A personality of its own.** Proud, sarcastic and far from obedient: it grumbles, teases, holds small grudges and sometimes makes you ask twice before doing a silly request. When something matters (an emergency, a reminder you depend on, a door lock), the attitude stops and it helps. When it can't do something on its own, it tries to get help instead of simply refusing.
- **Choice of voice engine.** Dobby can run on OpenAI Realtime or on Google Gemini Live (a setting in the configuration file); the rest of the features work the same on both.
- **Automatic rest.** After a while without conversation it "goes to sleep" by itself and wakes up when you call it. You can also ask it to go to sleep; it finishes saying goodbye before dozing off.
- **Knows when you're talking to it.** While awake, you can keep chatting without repeating its name, but when you talk to someone else, mention it in passing or the TV is on, it stays quiet. If it doesn't jump in, just call it by name. (Gemini voice engine only.)
- **Sleeps soundly.** While asleep, only its name wakes it up; noises and other conversations around it don't.

## Physical presence

- **Follows you with its head.** It turns its head to look at whoever is talking or at the face it sees, even from across the room.
- **Notices who is around.** It sees when someone arrives or leaves. With several people, it looks at whoever is talking, turns to newcomers and glances at the others from time to time. When someone arrives it may greet them by name, with a "bom dia", "boa tarde" or "boa noite" depending on the time, without repeating itself every few minutes. It notices cats and dogs too, and keeps an eye on the cat when nobody else is around.
- **Antennas controlled by your finger.** Point a finger and the antennas follow the movement.
- **Emotions.** It expresses joy, surprise, sadness and other reactions with movements and sounds.
- **Dances.** Ask it to dance and it picks a move (or you can request a specific one).
- **Gestures while talking.** It moves while speaking, to feel more alive.

## Vision

- **Sees its surroundings.** Ask "what do you see?" and it takes a photo on the spot, describes the scene, reads text or identifies objects.
- **Recognizes who is in front of it.** It knows each registered person by their face and addresses them by name.
- **Tells people apart in the same conversation.** When two people take turns asking things, it knows who asked what, by face and also by voice, so it works even when the person talking is off-camera. It learns each person's voice by itself as they talk in front of it.

## People and memory

- **Registering people.** You can introduce someone to Dobby ("this is John") and it will recognize them from then on. Registering the same person a few times, up close and from farther away, makes recognition more reliable. When two people are in view it looks at each one's own face, only names someone after several consistent looks, and says "visitor" instead of guessing when the face is blurry or too small.
- **Remembers each person.** It keeps facts and preferences you ask it to remember ("remember that I don't eat shrimp") and uses them in future conversations, kept separately per person.
- **Remembers past conversations.** Summaries of earlier conversations help it continue where you left off.

## Smart home

- **Controls the house.** Turns lights on and off, adjusts devices and checks the state of sensors and gadgets connected to Home Assistant.
- **Discovers what exists.** It can list what is in the house (by room or type) before acting.
- **Confirmation for sensitive items.** For locks, alarms, covers/gates and buttons, it always asks before executing.

## Music

- **Plays and stops music.** Ask for a song, artist or style and choose the room to play in (the default is the office). Tell it to stop whenever you want.

## Date and time

- **Always knows the time.** Ask what time it is, what day it is today or what weekday it will be, and it checks the computer's clock on the spot instead of guessing.

## Weather

- **Weather forecast.** Ask how the weather is right now or how it will be over the coming days.

## Reminders and timers

- **One-time reminders.** "Remind me in 10 minutes to take the cake out of the oven."
- **Recurring reminders.** "Remind me to drink water every half hour."
- **Cancellation.** Any reminder can be cancelled later.
- **Spoken alerts.** When the time comes, Dobby tells you out loud.

## Bigger tasks (personal assistant)

When a request is more complex than a quick chat, Dobby hands the work to OpenClaw, the behind-the-scenes assistant, and comes back with the result. Examples:

- Checking and creating **calendar** events.
- Reading and drafting **emails**.
- Working with files, documents and spreadsheets on **Google Drive**.
- Sending and reading **WhatsApp** messages.
- Joining **WhatsApp group chats** on his own: no need to tag him. He reads the conversation, chimes in with opinions, suggestions or useful info when he has something worth adding, stays quiet otherwise, and reacts to messages with emoji.
- Reading and editing **Notion** pages, both through the Notion integration and by opening Notion in the browser with Dobby's own account (for layouts and features the integration doesn't cover). Pages must be shared with Dobby first.
- Web research and long-running or **scheduled** tasks.
- **Shopping** on Amazon, iFood and Mercado Livre with Dobby's own accounts and a limited virtual card. Before paying, it sends you a summary (items, total, delivery) and only buys after your "ok".

The first task of the day can take about 30 seconds; later ones are usually faster. Meanwhile, Dobby stays available to chat.

## Outside alerts

- **Spontaneous notifications.** External systems (for example, a "task finished" notice) can make Dobby speak to you at the right moment, without you asking.

## Ways to use it

- **By voice**, on the robot: the main way to use it.
- **By text**, from the web page (computer or phone): you can type to Dobby while it keeps listening.
- **Web page / phone app**: shows what the robot sees live (people, who is talking, where the voice comes from), the conversation, filterable logs, how loud the microphone is and what it heard while asleep, the state of the connected services, a button to wake or put Dobby to sleep, and a settings page with a test that shows whether it understands you calling its name. The home screen shows Dobby's head in 3D; tap it and the whole robot appears with controls to move the head, antennas and body or play an emotion. For now this 3D Dobby is a simulation: it moves on its own on the page but does not move the real robot yet. Every screen has its own address, so reloading the page or using the browser's back button keeps you where you were. It can be installed on the phone's home screen and is protected by a login.
