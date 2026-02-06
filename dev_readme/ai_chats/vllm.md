https://docs.vllm.ai/en/latest/features/structured_outputs/#online-serving-openai-api
    prompt_embeds: bytes | list[bytes] | None = None
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."
        ),
    )
    response_format: AnyResponseFormat | None = Field(
        default=None,
        description=(
            "Similar to chat completion, this parameter specifies the format "
            "of output. Only {'type': 'json_object'}, {'type': 'json_schema'}"
            ", {'type': 'structural_tag'}, or {'type': 'text' } is supported."
        ),
    )
    structured_outputs: StructuredOutputsParams | None = Field(
        default=None,
        description="Additional kwargs for structured outputs",
    )
    guided_json: str | dict | BaseModel | None = Field(
        default=None,
        description=(
            "`guided_json` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `json` to `structured_outputs` instead."
        ),
    )
    guided_regex: str | None = Field(
        default=None,
        description=(
            "`guided_regex` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `regex` to `structured_outputs` instead."
        ),
    )
    guided_choice: list[str] | None = Field(
        default=None,
        description=(
            "`guided_choice` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `choice` to `structured_outputs` instead."
        ),
    )
    guided_grammar: str | None = Field(
        default=None,
        description=(
            "`guided_grammar` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `grammar` to `structured_outputs` instead."
        ),
    )
    structural_tag: str | None = Field(
        default=None,
        description=("If specified, the output will follow the structural tag schema."),
    )
    guided_decoding_backend: str | None = Field(
        default=None,
        description=(
            "`guided_decoding_backend` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please remove it from your request."
        ),
    )
    guided_whitespace_pattern: str | None = Field(
        default=None,
        description=(
            "`guided_whitespace_pattern` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `whitespace_pattern` to `structured_outputs` instead."
        ),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    logits_processors: LogitsProcessors | None = Field(
        default=None,
        description=(
            "A list of either qualified names of logits processors, or "
            "constructor objects, to apply when sampling. A constructor is "
            "a JSON object with a required 'qualname' field specifying the "
            "qualified name of the processor class/factory, and optional "
            "'args' and 'kwargs' fields containing positional and keyword "
            "arguments. For example: {'qualname': "
            "'my_module.MyLogitsProcessor', 'args': [1, 2], 'kwargs': "
            "{'param': 'value'}}."
        ),
    )

    return_tokens_as_token_ids: bool | None = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."
        ),
    )
    return_token_ids: bool | None = Field(
        default=None,
        description=(
            "If specified, the result will include token IDs alongside the "
            "generated text. In streaming mode, prompt_token_ids is included "
            "only in the first chunk, and token_ids contains the delta tokens "
            "for each chunk. This is useful for debugging or when you "
            "need to map generated text back to input tokens."
        ),
    )

    cache_salt: str | None = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit). Not supported by vLLM engine V0."
        ),
    )

    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )

    vllm_xargs: dict[str, str | int | float] | None = Field(
        default=None,
        description=(
            "Additional request parameters with string or "
            "numeric values, used by custom extensions."
        ),
    )