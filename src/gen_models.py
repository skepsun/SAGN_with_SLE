from models import SAGN, PlainSAGN, SimpleSAGN, LPMLP, LPSIGN

def get_model(in_feats, n_classes, stage, args):
    num_hops = args.K + 1
    if args.dataset == "ogbn-mag":
        label_in_feats = n_classes
    else:
        label_in_feats = n_classes
    use_labels = args.use_labels and ((not args.inductive) or stage > 0)
    use_features = not args.avoid_features
    if args.model == "sagn":
        model = SAGN(in_feats, args.num_hidden, n_classes, label_in_feats, num_hops,
                        args.mlp_layer, args.num_heads, 
                        dropout=args.dropout, 
                        input_drop=args.input_drop, 
                        attn_drop=args.attn_drop, 
                        use_labels=use_labels,
                        use_features=use_features)
    
    if args.model == "mlp":
        model = LPMLP(in_feats, args.num_hidden, n_classes,
                 args.mlp_layer,  
                 args.dropout, 
                 bias=True,
                 residual=False,
                 input_drop=args.input_drop, 
                 use_labels=use_labels)
    
    if args.model == "simple_sagn":
        model = SimpleSAGN(in_feats, args.num_hidden, n_classes, label_in_feats, num_hops,
                        args.mlp_layer, args.num_heads, 
                        weight_style=args.weight_style,
                        dropout=args.dropout, 
                        input_drop=args.input_drop, 
                        attn_drop=args.attn_drop, 
                        use_labels=use_labels,
                        use_features=use_features)
    
    if args.model == "plain_sagn":
        model = PlainSAGN(in_feats, args.num_hidden, n_classes, label_in_feats,
                        args.mlp_layer, args.num_heads, 
                        dropout=args.dropout, 
                        input_drop=args.input_drop, 
                        attn_drop=args.attn_drop,
                        use_labels=use_labels,
                        use_features=use_features)
    
    if args.model == "sign":
        model = LPSIGN(in_feats, args.num_hidden, n_classes, label_in_feats, num_hops,
                args.mlp_layer, 
                dropout=args.dropout, 
                input_drop=args.input_drop, 
                use_labels=use_labels)

    return model