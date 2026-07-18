use oido_core::phrase_filter::is_repetition_loop;

fn main() {
    let cases = [
        ("hola×10", "hola hola hola hola hola hola hola hola hola hola"),
        ("sub×7",   "subscribe subscribe subscribe subscribe subscribe subscribe subscribe"),
        ("user",    "Y yo, en español, me voy a decir que me voy a decir que me voy a decir que me voy a decir que me voy a decir que"),
    ];
    for (name, text) in cases {
        let wcount = text.split_whitespace().count();
        let result = is_repetition_loop(text);
        println!("{name}: words={wcount} → is_repetition_loop={result}");
    }
}
