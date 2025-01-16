import os


def add_shared_arguments(parser):
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--data_dir", type=str, default="eval_data", help="Path to personabench datasets")
    parser.add_argument(
        '--model', type=str, default="gpt-4o",
        help=(
            'Only support OpenAI models for now. The model to use (e.g., "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", '
            '"gpt-4-32k-0314", "gpt-4-turbo-2024-04-09", "gpt-4-0613", '
            '"gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo-1106" ...)'
        )
    )
    parser.add_argument('--api_key', type=str, default=os.getenv('OPENAI_API_KEY'), help='Better set from environment variable "OPAI_API_KEY"')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser