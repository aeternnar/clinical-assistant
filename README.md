# Clinical Assistant

![cover](assets/cover.jpg)

## Описание

Проект относится к области Natural Language Processing (обработка естественного языка).  
В основе проекта лежит модель [Gemma](https://ai.google.dev/gemma) — LLM от компании Google.
Благодаря дополнительному обучению на медицинском корпусе, модель была адаптирована под задачу генерации ответов на медицинские вопросы.

Модель интегрирована в web-приложение через backend-сервис на **FastAPI**.  
Пользовательский интерфейс позволяет задать вопрос через форму и получить ответ в реальном времени.  

Разработанный ассистент может быть полезен:
- студентам — для подготовки и изучения медицинских дисциплин,  
- врачам и исследователям — для оперативного поиска справочной информации,  
- широкой аудитории — для повышения уровня медицинской грамотности.  

## Данные

Для обучения использовался корпус текстовых данных [Healthcare NLP](https://www.kaggle.com/datasets/jpmiller/layoutlm), содержащий пары «вопрос–ответ».  
В датасете содержатся более 24 тысяч ответов на вопросы, охватывающих различные области медицины.  

Каждая запись в наборе данных приведена к требуемому формату с использованием специальных токенов, указывающих границы между вопросами пользователя и ответами нейросети:

```text
<start_of_turn>user
How is high blood pressure diagnosed?<end_of_turn>
<start_of_turn>model
The only way of knowing whether you have high blood pressure is to have a blood pressure test, which can be done at your GP surgery, some pharmacies, as part of your NHS Health Check, or in some workplaces. All adults over 40 are advised to have their blood pressure checked at least every 5 years.<end_of_turn>

<start_of_turn>user
What are the lifestyle changes that can help reduce blood pressure?<end_of_turn>
<start_of_turn>model
Lifestyle changes that can help prevent and lower high blood pressure include reducing the amount of salt you eat and having a generally healthy diet, cutting back on alcohol, losing weight if you're overweight, exercising regularly, cutting down on caffeine, and stopping smoking.<end_of_turn>

<start_of_turn>user
What are the medicines used to control high blood pressure?<end_of_turn>
<start_of_turn>model
Common blood pressure medicines include ACE inhibitors, angiotensin-2 receptor blockers (ARBs), calcium channel blockers, diuretics, beta blockers, alpha blockers, and other diuretics. The medicine recommended for you will depend on factors like how high your blood pressure is, your age, and your ethnicity.<end_of_turn>
```

## Модель

Для ассистента использовалась [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b) — компактная LLM decoder-архитектуры от Google DeepMind с 2,6 млрд параметров, оптимизированная для работы на устройствах с ограниченными ресурсами.

Она была дообучена с использованием **LoRA** и **4-битной квантизации**, что позволило существенно снизить потребление памяти и вычислительных ресурсов при сохранении качества генерации ответов.

## Результаты и метрики

Качество модели оценивалось на тестовом наборе данных. Ниже представлены основные метрики:

| Метрика              | Значение  |
|----------------------|-----------|
| **SacreBLEU**        | 13.82     |
| **ROUGE-1**          | 0.4173    |
| **ROUGE-2**          | 0.2115    |
| **ROUGE-L**          | 0.3427    |
| **ROUGE-Lsum**       | 0.3426    |
| **BERTScore Precision**      | 0.9192    |
| **BERTScore Recall**      | 0.9020    |
| **BERTScore F1**     | 0.9103    |

Модель показывает умеренное совпадение с эталонными ответами по метрикам BLEU и ROUGE, что объясняется вариативностью формулировок в естественном языке, при этом высокие значения BERTScore подтверждают, что ответы ассистента сохраняют смысловую близость к референсу, что особенно важно в медицинском контексте.

### Примеры работы
![QA](assets/test_gif.gif)
