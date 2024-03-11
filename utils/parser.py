import argparse

def set_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=7136, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--date', default='0313', type=str)

    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    parser.add_argument('--model', default='Proposed', type=str)

    parser.add_argument('--window_min', default=-20, type=int)
    parser.add_argument('--window_max', default=100, type=int)
    parser.add_argument('--crop_size', default=128, type=int)
    parser.add_argument('--filter_threshold', default=0.4, type=float)
    parser.add_argument('--gaussian_filter', default=1, type=int)

    return parser.parse_args()