import torch
import logging

def decode_sdf(decoder, latent_vector, queries, atc_vec=None, do_sup_with_part=False, specs=None):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_vecs = latent_vector.expand(num_samples, -1)
        if atc_vec is not None:
            atc_vecs = atc_vec.expand(num_samples, -1).cuda()
            inputs = torch.cat([latent_vecs, queries, atc_vecs], 1)

        else:
            inputs = torch.cat([latent_vecs, queries], 1)

        if do_sup_with_part:
            sdf, _ = decoder(inputs)
        else:
            sdf = decoder(inputs)

    return sdf


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)
